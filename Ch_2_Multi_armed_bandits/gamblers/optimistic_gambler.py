import numpy as np
from .eps_greedy_gambler import EpsilonGreedyGambler


class OptimisticGambler(EpsilonGreedyGambler):
    def __init__(self, n: int = 10, epsilon: float = 0.0, alpha: float = 0.1, init: float = 10.0):
        self.init = init
        super().__init__(n, epsilon, alpha)

    def reset(self) -> None:
        self.Q = np.ones([self.n]) * self.init
        