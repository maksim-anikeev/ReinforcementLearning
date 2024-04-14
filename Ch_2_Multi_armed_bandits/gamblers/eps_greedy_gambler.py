import numpy as np
from .gambler import Gambler


class EpsilonGreedyGambler(Gambler):
    def __init__(self, n: int = 10, epsilon: float = 0.1, alpha: float = 0.1):
        super().__init__()
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = None
        self.reset()

    def arm(self) -> None:
        if self.np_random.rand() < self.epsilon:
            arm = self.np_random.randint(0, self.n)
        else:
            arm = np.argmax(self.Q)
        return arm
    
    def update(self, arm: int, reward: float) -> None:
        self.Q[arm] += self.alpha * (reward - self.Q[arm])

    def reset(self) -> None:
        self.Q = self.np_random.rand(self.n) * 1e-5
        