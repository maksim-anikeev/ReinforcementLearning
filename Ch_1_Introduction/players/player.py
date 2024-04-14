from environment import env
from state import State


class Player:
    def __init__(self, **kwargs):
        self.env = env
        self.symbol = None

    def reset(self):
        pass

    def set_state(self, state: State):
        raise NotImplementedError
        
    def set_symbol(self, symbol: int):
        self.symbol = symbol

    def act(self):
        raise NotImplementedError

    def backup(self):
        raise NotImplementedError

    def save_policy(self):
        raise NotImplementedError

    def load_policy(self):
        raise NotImplementedError
