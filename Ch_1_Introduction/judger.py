from environment import env
from players import Player
from state import State

class Judger:
    def __init__(self, player1: Player, player2: Player):
        self._env = env
        self.p1 = player1
        self.p2 = player2
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def play(self, print_state: bool = False) -> int:
        alternator = self.alternate()
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)

        if print_state:
            current_state.print_state()

        while True:
            current_player = next(alternator)
            i, j, symbol = current_player.act()
            next_state_hash = current_state.next_state(i, j, symbol).hash
            current_state = self._env.get_state(next_state_hash)
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if print_state:
                current_state.print_state()
            if current_state.is_end:
                return current_state.winner
