import pickle
import numpy as np

from .player import Player
from state import State

from typing import List, Tuple


class AIPlayer(Player):
    def __init__(self, step_size: float = 0.1, epsilon: float = 0.1):
        super().__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.estimations = dict()

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state: State):
        self.states.append(state)
        self.greedy.append(True)
        
    def set_symbol(self, symbol: int):
        super().set_symbol(symbol)
        for hash in self.env.all_states:
            state = self.env.get_state(hash)
            if state.is_end:
                if state.winner == self.symbol:
                    self.estimations[hash] = 1.0
                elif state.winner == 0:
                    self.estimations[hash] = 0.5
                else:
                    self.estimations[hash] = 0
            else:
                self.estimations[hash] = 0.5

    def act(self) -> List[int]:
        state = self.states[-1]
        possible_states = []
        possible_positions = []
        for i in range(state.board_size):
            for j in range(state.board_size):
                if state.board[i, j] == 0:
                    possible_positions.append((i, j))
                    possible_states.append(state.next_state(i, j, self.symbol).hash)
        
        if np.random.rand() < self.epsilon:
            return self.make_exploratory_action(possible_positions)
        else:
            return self.make_greedy_action(possible_positions, possible_states)
    
    def make_exploratory_action(self, possible_positions: List[Tuple[int]]) -> List[int]:
        random_index = np.random.randint(len(possible_positions))
        action = list(possible_positions[random_index])
        action.append(self.symbol)
        self.greedy[-1] = False
        return action
    
    def make_greedy_action(self, possible_positions: List[Tuple[int]], possible_states: List[State]) -> List[int]:
        best_value = -np.inf
        best_action = None
        for action, hash in zip(possible_positions, possible_states):
            value = self.estimations[hash]
            # print(hash, value)
            if ((value > best_value) or
                (value == best_value and np.random.rand() < 0.5)):
                best_value = value
                best_action = action
        
        # print(best_action, best_value)
        action = list(best_action)
        action.append(self.symbol)
        return action

    def backup(self):
        hashes = [state.hash for state in self.states]
        for i in reversed(range(len(hashes) - 1)):
            hash = hashes[i]
            td_error = self.greedy[i] * (
                self.estimations[hashes[i + 1]] - self.estimations[hash]
            )
            self.estimations[hash] += self.step_size * td_error

    def save_policy(self):
        with open(f'policy_{"first" if self.symbol == 1 else "second"}', 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open(f'policy_{"first" if self.symbol == 1 else "second"}', 'rb') as f:
            self.estimations = pickle.load(f)
