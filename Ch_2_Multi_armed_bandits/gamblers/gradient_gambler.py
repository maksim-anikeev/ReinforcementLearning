import numpy as np
from .gambler import Gambler


class GradientGambler(Gambler):
    def __init__(self, n: int = 10, alpha: float = 0.1):
        super().__init__()
        self.n = n
        self.alpha = alpha
        self.last_p = None
        self.H = None
        self.R = None
        self.reset()

    @property
    def p(self) -> float:
        exps = np.exp(self.H - np.max(self.H))
        p = exps / exps.sum()
        self.last_p = p
        return p
    
    def arm(self) -> int:
        arms = [arm for arm in range(self.n)]
        arm = self.np_random.choice(arms, p=self.p)
        return arm
    
    def update(self, arm: int, reward: float) -> None:
        others = [a for a in range(self.n) if a != arm]
        p = self.last_p
        self.H[arm] += self.alpha * (reward - self.R) * (1 - p[arm])
        self.H[others] -= self.alpha * (reward - self.R) * p[others]
        self.R += self.alpha * (reward - self.R)
    
    def reset(self) -> None:
        self.last_p = None
        self.H = self.np_random.rand(self.n) * 1e-2
        self.R = 0.0