import numpy as np
from .gambler import Gambler


class UCBGambler(Gambler):
    def __init__(self, n: int = 10, c: float = 1.0):
        super().__init__()
        self.n = n
        self.c = c
        self.t = 1
        self.N = None
        self.Q = None
        self.reset()

    def arm(self) -> None:
        ucb = self.Q + self.c * np.sqrt(np.log(self.t) / (self.N + 1e-5))
        arm = np.argmax(ucb)
        return arm
    
    def update(self, arm: int, reward: float) -> None:
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]
        self.t += 1

    def reset(self) -> None:
        self.N = np.zeros([self.n])
        self.Q = self.np_random.rand(self.n) * 1e-5
        