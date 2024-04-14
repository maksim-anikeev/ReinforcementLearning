from state import State

from .player import Player

from typing import List


class HumanPlayer(Player):
    def __init__(self, **kwargs):
        super().__init__()
        self.keys = ['7', '8', '9', '4', '5', '6', '1', '2', '3']
        self.state = None

    def set_state(self, state: State):
        self.state = state

    def act(self) -> List[int]:
        self.state.print_state()
        key = input("Input your position:")
        index = self.keys.index(key)
        i = index // self.state.board_size
        j = index % self.state.board_size
        return i, j, self.symbol