"""
Implementation with a payoff matrix for the population, using a fixed set of openings for each generation.
The fitness of an individual is its payoff against the Nash mixture of the current population.
At the moment, this approach beahaves worse than simple GP, maybe there aren't enough generations.
"""
import numpy as np
import argparse
import random
import multiprocessing
import scipy
import pickle

from othello_rules.bitboard_utils import (count_bits, get_moves_index,
                                          get_player_board, make_move,
                                          possible_moves)

from first_start import GPAgent, RandomAgent, toolbox, win_rate, algorithms, tools, OthelloGui, HumanPlayer

FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
INIT = (np.uint64(0x0000000810000000), np.uint64(0x0000001008000000))


def legal_moves(board, pid):
    own, opp = get_player_board(board, pid)
    return get_moves_index(possible_moves(own, opp, FULL ^ (own | opp)))


def enumerate_openings(depth=4):
    """Tutte le sequenze di `depth` mosse dalla posizione iniziale.
    depth=4 -> 244 aperture (nessun pass entro 4 plies; non usare depth grandi)."""
    out = []

    def rec(board, turn, seq):
        if len(seq) == depth:
            out.append(tuple(seq))
            return
        for m in legal_moves(board, turn):
            rec(make_move(board, int(m), turn), 3 - turn, seq + [int(m)])

    rec(INIT, 1, [])
    return out


def play_match(agent_black, agent_white, opening):
    black, white = agent_black.copy(), agent_white.copy()
    black.set_id(1)
    white.set_id(2)
    players = {1: black, 2: white}

    board, turn, passes = INIT, 1, 0
    for m in opening:
        board = make_move(board, m, turn)
        turn = 3 - turn
    while passes < 2:
        move = players[turn].get_move(board)
        if move is None:                     # nessuna mossa: pass
            passes += 1
            turn = 3 - turn
            continue
        passes = 0
        board = make_move(board, int(move), turn)
        turn = 3 - turn

    b, w = count_bits(board[0]), count_bits(board[1])
    return 1.0 if b > w else (0.5 if b == w else 0.0)


def payoff(agent_a, agent_b, openings):
    tot = 0.0
    for op in openings:
        tot += play_match(agent_a, agent_b, op)           # a black
        tot += 1.0 - play_match(agent_b, agent_a, op)     # a white
    return 2.0 * tot / (2 * len(openings)) - 1.0


def payoff_matrix(agents, openings):
    """Matrice n x n; si calcola solo il triangolo superiore."""
    n = len(agents)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = payoff(agents[i], agents[j], openings)
            A[j, i] = -A[i, j]
    return A


def population_fitness(A):
    """F_i = payoff di i contro la miscela Nash p* della popolazione corrente.
    Sempre <= 0 (il valore del gioco e' 0); = 0 esattamente sul supporto."""
    p, v = solve_zero_sum_row(A)
    assert abs(v) < 1e-6
    p = np.clip(p, 0.0, None)
    p /= p.sum()
    return A @ p


def solve_zero_sum_row(A: np.ndarray):
    """Trova la strategia mista p ottima per il giocatore Riga (payoff A)
    e il valore del gioco v in un gioco a somma zero."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    # Variables: x = [p_0, p_1, ..., p_{m-1}, v]
    # Objective: minimize -v
    c = np.zeros(m + 1)
    c[-1] = -1.0

    # -A^T p + v <= 0  <=>  A^T p >= v
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    b_ub = np.zeros(n)

    # sum(p_i) = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    # p_i >= 0, v whatever
    bounds = [(0.0, 1.0)] * m + [(None, None)]

    res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Ottimizzazione fallita: {res.message}")

    p = res.x[:m]
    v = res.x[-1]
    return p, v

 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--ngen", type=int, default=30)
    ap.add_argument("--kopen", type=int, default=6, help="aperture per generazione (2 partite ciascuna)")
    ap.add_argument("--heldout", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cxpb", type=float, default=0.85)
    ap.add_argument("--mutpb", type=float, default=0.15)
    ap.add_argument("--elite", type=int, default=2)
    args = ap.parse_args()
 
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    all_openings = enumerate_openings(4)                       
    heldout_seeds = rng.integers(2**31, size=args.heldout)
 
    population = toolbox.population(n=args.pop)
    for gen in range(args.ngen + 1):
        idx = rng.choice(len(all_openings), size=args.kopen, replace=False)
        openings = [all_openings[i] for i in idx]
 
        agents = [GPAgent(toolbox.compile(expr=ind)) for ind in population]
        A = payoff_matrix(agents, openings)
        for ind, f in zip(population, population_fitness(A)):
            ind.fitness.values = (float(f),)
 
        best = tools.selBest(population, 1)[0]
        msg = (f"gen {gen:3d}  fitness best {best.fitness.values[0]:+.3f}  "
               f"size {np.mean([len(i) for i in population]):.1f}")
        if gen % 5 == 0 or gen == args.ngen:
            msg += f"  | best vs random {win_rate(best, heldout_seeds):.3f}"
        print(msg, flush=True)
        if gen == args.ngen:
            break
 
        elite = [toolbox.clone(i) for i in tools.selBest(population, args.elite)]
        selected = toolbox.select(population, len(population) - args.elite)
        offspring = algorithms.varAnd([toolbox.clone(i) for i in selected],
                                      toolbox, args.cxpb, args.mutpb)
        population = elite + offspring
 
    print("\nmiglior individuo:", str(best))

    with open("best_individual_Nash_GP.pkl", "wb") as f:
            pickle.dump(best, f)

    gui = OthelloGui(player1=GPAgent(toolbox.compile(expr=best)), player2=HumanPlayer())
    gui.run_game()
 
if __name__ == "__main__":
    main()