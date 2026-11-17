"""
First implementation of the code, just a basic GP agent against a random agent.
Are used the most of the already existing functions and classes.
Have to check if there are some improvements
"""

import argparse
import functools
import multiprocessing
import operator
import random

import numpy as np
from deap import algorithms, base, creator, gp, tools

from othello_rules.agents import Agent, RandomAgent
from othello_rules.othello_simulations import OthelloSimulation

from othello_rules.bitboard_utils import (
    count_bits,
    get_player_board,
    make_move,
    possible_moves,
    get_moves_index,
)


# GP features

FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
CORNERS = np.uint64(0x8100000000000081)

FEATURES = ["diff", "mob", "omob", "corner", "xsq", "empt", "cdiff"]
N_FEAT = len(FEATURES)


def _move_bit(move):
    """Return the uint64 bit corresponding to a square index."""
    return np.uint64(1) << np.uint64(move)


def features(board, pid, moves):
    """Build one feature row per candidate move, evaluated after that move."""
    result = np.zeros((len(moves), N_FEAT), dtype=np.float64)

    for i, move in enumerate(moves):
        next_board = make_move(board, int(move), pid)
        own, opp = get_player_board(next_board, pid)
        empty = FULL ^ (own | opp)

        n_own = count_bits(own)
        n_opp = count_bits(opp)

        result[i, 0] = n_own - n_opp
        result[i, 1] = count_bits(possible_moves(own, opp, empty))
        result[i, 2] = count_bits(possible_moves(opp, own, empty))
        result[i, 3] = float(int(move) in (0, 7, 56, 63))
        result[i, 4] = float(int(move) in (9, 14, 49, 54))
        result[i, 5] = 64 - n_own - n_opp
        result[i, 6] = count_bits(own & CORNERS) - count_bits(opp & CORNERS)

    return result


def score_moves(func, feature_matrix):
    """Evaluate the GP expression for every legal move and sanitize scores."""
    scores = func(*feature_matrix.T)
    scores = np.broadcast_to(
        np.asarray(scores, dtype=float),
        (feature_matrix.shape[0],),
    )
    return np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)


# GP agent and play function

class GPAgent(Agent):

    def __init__(self, gp_function):
        super().__init__()
        self.gp_function = gp_function

    def get_move(self, board, events=None):

        own, opp = get_player_board(board, self.id)
        empty = np.uint64(0xFFFFFFFFFFFFFFFF) ^ (own | opp)

        moves_bb = possible_moves(own, opp, empty)
        moves = get_moves_index(moves_bb)

        if len(moves) == 0:
            return None

        feature_matrix = features(board, self.id, moves)
        scores = score_moves(self.gp_function, feature_matrix)

        return int(moves[np.argmax(scores)])

    def copy(self):
        return GPAgent(self.gp_function)


def play(func, agent_id, seed, n_open=4):
    gp_agent = GPAgent(func)
    random_agent = RandomAgent()

    winner = OthelloSimulation.simulate_game(
        (gp_agent, random_agent)
    )

    if winner == gp_agent.id:
        return 1.0

    if winner == 0:
        return 0.5

    return 0.0


# Genetic Programming setup

def protected_division(a, b):
    """Element-wise protected division; return 1 near a zero denominator."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.abs(b) > 1e-6
    return np.where(valid, a / np.where(valid, b, 1.0), 1.0)


pset = gp.PrimitiveSet("MAIN", N_FEAT)
for i, name in enumerate(FEATURES):
    pset.renameArguments(**{f"ARG{i}": name})

for primitive, name in (
    (np.add, "add"),
    (np.subtract, "sub"),
    (np.multiply, "mul"),
    (protected_division, "pdiv"),
    (np.maximum, "max"),
    (np.minimum, "min"),
):
    pset.addPrimitive(primitive, 2, name=name)

pset.addPrimitive(np.negative, 1, name="neg")
pset.addEphemeralConstant(
    "c",
    functools.partial(random.uniform, -1.0, 1.0),
)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate(
    "mate",
    gp.staticLimit(key=operator.attrgetter("height"), max_value=8),
)
toolbox.decorate(
    "mutate",
    gp.staticLimit(key=operator.attrgetter("height"), max_value=8),
)


# Fitness

def win_rate(individual, seeds):
    func = toolbox.compile(expr=individual)
    total = 0.0

    for seed in seeds:
        seed = int(seed)
        total += play(func, 1, seed)
        total += play(func, 2, seed)

    return total / (2 * len(seeds))


def evaluate(individual, seeds):
    return (win_rate(individual, seeds),)


# Evolution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--ngen", type=int, default=30)
    parser.add_argument(
        "--pairs",
        type=int,
        default=12,
        help="Seeds per generation (2 games per seed).",
    )
    parser.add_argument("--heldout", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cxpb", type=float, default=0.85)
    parser.add_argument("--mutpb", type=float, default=0.15)
    parser.add_argument("--elite", type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    master_rng = np.random.default_rng(args.seed)
    heldout_seeds = master_rng.integers(2**31, size=args.heldout) + 2**31

    pool = None
    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers)
        toolbox.register("map", pool.map)

    try:
        population = toolbox.population(n=args.pop)

        for generation in range(args.ngen + 1):
            # Common random numbers: all individuals share seeds per generation.
            seeds = master_rng.integers(2**31, size=args.pairs)
            fitnesses = toolbox.map(
                functools.partial(evaluate, seeds=seeds),
                population,
            )

            for individual, fitness in zip(population, fitnesses):
                individual.fitness.values = fitness

            best = tools.selBest(population, 1)[0]
            values = np.array(
                [individual.fitness.values[0] for individual in population]
            )

            message = (
                f"gen {generation:3d}  "
                f"train mean {values.mean():.3f} max {values.max():.3f}  "
                f"size {np.mean([len(ind) for ind in population]):.1f}"
            )

            if generation % 5 == 0 or generation == args.ngen:
                heldout = win_rate(best, heldout_seeds)
                se = np.sqrt(
                    max(heldout * (1 - heldout), 1e-9)
                    / (2 * len(heldout_seeds))
                )
                message += f"  | held-out best {heldout:.3f} +/- {se:.3f}"

            print(message, flush=True)

            if generation == args.ngen:
                break

            elite = [
                toolbox.clone(ind)
                for ind in tools.selBest(population, args.elite)
            ]
            selected = toolbox.select(population, len(population) - args.elite)
            offspring = algorithms.varAnd(
                [toolbox.clone(ind) for ind in selected],
                toolbox,
                args.cxpb,
                args.mutpb,
            )
            population = elite + offspring

        print("\nMiglior individuo (ultima gen.):", str(best))

    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
