from state import State

from typing import Dict


entity = None


class Environment:
    def __init__(self):        
        self.all_states = dict()
        self._generate_all_states()
    
    def get_state(self, hash: int) -> State:
        return self.all_states[hash]

    def _generate_all_states(self):
        current_symbol = 1
        current_state = State()
        self.all_states[current_state.hash] = current_state
        self._generate_all_states_impl(current_state, current_symbol)

    def _generate_all_states_impl(self, current_state: State, current_symbol: int):
        board_size = current_state.board_size
        for i in range(board_size):
            for j in range(board_size):
                if current_state.board[i][j] == 0:
                    new_state = current_state.next_state(i, j, current_symbol)
                    new_hash = new_state.hash
                    if new_hash not in self.all_states:
                        self.all_states[new_hash] = new_state
                        if not new_state.is_end:
                            self._generate_all_states_impl(new_state, -current_symbol)



env = Environment()