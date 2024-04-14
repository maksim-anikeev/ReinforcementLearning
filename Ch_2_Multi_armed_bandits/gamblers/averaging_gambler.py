import numpy as np
from .gambler import Gambler


class AveragingGambler(Gambler):
    def __init__(self, n: int = 10, epsilon: float = 0.1):
        super().__init__()
        self.n = n
        self.epsilon = epsilon
        self.Q = None
        self.reset()

    def arm(self) -> int:
        if self.np_random.rand() < self.epsilon:
            arm = self.np_random.randint(0, self.n)
        else:
            arm = np.argmax(self.Q)
        return arm
    
    def update(self, arm: int, reward: float) -> None:
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

    def reset(self):
        self.N = np.zeros([self.n])
        self.Q = self.np_random.rand(self.n) * 1e-5
