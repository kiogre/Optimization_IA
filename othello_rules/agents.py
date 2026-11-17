from abc import ABC, abstractmethod
from os import environ
import random
import time

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from pygame import MOUSEBUTTONDOWN

from .bitboard_utils import get_moves_index, possible_moves, get_player_board

SCREEN_WIDTH = SCREEN_HEIGHT = 800
CELL_SIZE = SCREEN_HEIGHT // 8

class Player(ABC):
    """
    Abstract base class for a player in the Othello game.

    Attributes:
        id (int): The identifier for the player, either 1 or 2.
    """
    def __init__(self):
        self.id = None
    
    def __repr__(self):
        """
        Returns a string representation of the Player object.

        Returns:
            str: A string representation including the player's class name and ID.
        """
        class_name = self.__class__.__name__
        return f"{class_name}(id={self.id})"
        
    def set_id(self, player_id):
        """
        Sets the player's ID.

        Parameters:
            player_id (int): The ID to be set for the player.
        """
        if player_id != 1 and player_id != 2:
            raise ValueError("Player ID must be either 1 or 2.")
        
        self.id = player_id

    @abstractmethod
    def get_move(self, board, events):
        """
        Abstract method to be implemented by subclasses to get the player's move.

        Parameters:
            board (UniTuple(uint64, 2)): The current state of the game board.
            events (list of pygame.event.Event): A list of Pygame events.

        Returns:
            int: The chosen move as a bitboard index, or None if no valid move is chosen.
        """
        raise NotImplementedError('You must implement get_move()')
    
class Agent(Player):
    """
    Class for a AI Agent in the Othello game.
    """
    @abstractmethod
    def copy(self):
        """
        Returns a new instance of the Player with the same parameters as the current instance.
        Used for simulations.

        Returns:
        Agent: A new instance with the same parameters.
        """
        raise NotImplementedError('You must implement copy()') 
    
    
class HumanPlayer(Player):
    """
    Class for a human player in the Othello game.
    """
    def get_move(self, board, events):
        """
        Gets the move from the human player based on mouse input.

        Parameters:
            board (UniTuple(uint64, 2)): The current state of the game board.
            events (list of pygame.event.Event): A list of Pygame events.

        Returns:
            int: The chosen move as a bitboard index, or None if no valid move is chosen.
        """
        
        player_bb, opponent_bb = get_player_board(board, self.id)
        
        empty_squares = (player_bb | opponent_bb) ^ 0xFFFFFFFFFFFFFFFF
        possible_moves_bb_player = possible_moves(player_bb, opponent_bb, empty_squares)
        valid_moves = get_moves_index(possible_moves_bb_player)
        
        for event in events:
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                row = y // CELL_SIZE
                col = x // CELL_SIZE
                
                bb_indice = row * 8 + col
                if bb_indice in valid_moves:
                    return bb_indice
        return None
    
class RandomAgent(Agent):
    """
    Class for a Random Agent in the Othello game.
    """
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
    
    def get_move(self, board, events):
        """
        Gets a random move from the possible moves.

        Parameters:
            board (UniTuple(uint64, 2)): The current state of the game board.
            events (list of pygame.event.Event): A list of Pygame events.

        Returns:
            int: The chosen move as a bitboard index, or None if no valid move is chosen.
        """
        player_bb, opponent_bb = get_player_board(board, self.id)
        
        empty_squares = (player_bb | opponent_bb) ^ 0xFFFFFFFFFFFFFFFF
        possible_moves_bb_player = possible_moves(player_bb, opponent_bb, empty_squares)
        valid_moves = get_moves_index(possible_moves_bb_player)
        move = None
        
        if valid_moves.shape[0] > 0:
            move = random.choice(valid_moves)
            if self.verbose == True:
                print(f"Player {self.id} --> {move}")
            
        return move
    
    def copy(self):
        """
        Returns a new instance of RandomAgent.

        Returns:
        RandomAgent: A new instance with the same parameters.
        """
        return RandomAgent(verbose=self.verbose)