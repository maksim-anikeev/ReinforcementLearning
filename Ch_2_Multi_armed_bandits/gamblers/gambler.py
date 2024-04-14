import numpy as np


class Gambler:
    def __init__(self) -> None:
        self.np_random = None
        self.seed()

    def seed(self, seed: int = None) -> None:
        self.np_random = np.random.RandomState(seed)

    def arm(self):
        raise NotImplementedError
    
    def update(self, arm: int, reward: float) -> None:
        pass

    def reset(self) -> None:
        pass