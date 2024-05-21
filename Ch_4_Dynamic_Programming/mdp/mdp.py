import numpy as np
from gymnasium import Env


class MDP(Env):
    def __init__(self) -> None:
        self.state = None
        self.terminals = []

    def transitions(self, state, action: int) -> None:
        raise NotImplementedError()

    def step(self, action: int) -> None:
        transitions, rewards, probs, terminals = self.transitions(self.state, action)
        action_index = np.random.choice(probs.size, p=probs)
        self.state = transitions[action_index]
        return self.state, rewards[action_index], terminals[action_index], {}

    def render(self) -> None:
        print(self.state)
