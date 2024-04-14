from .gambler import Gambler


class RandomGambler(Gambler):
    def __init__(self, n: int = 10):
        super().__init__()
        self.n = n

    def arm(self) -> int:
        return self.np_random.choice(self.n)