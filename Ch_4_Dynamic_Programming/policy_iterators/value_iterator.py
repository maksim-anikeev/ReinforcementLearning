import numpy as np
from typing import Tuple

from mdp.mdp import MDP
from .policy_iterator import PolicyIterator


class ValueIterator(PolicyIterator):
    def __init__(self, mdp: MDP, tolerance=1e-6) -> None:
        super().__init__(mdp)
        self.tolerance = tolerance

    def find_optimal_policy(self) -> None:
        old_value = 2 * self.tolerance
        self.value = np.zeros(shape=self.mdp.observation_space.nvec)
        while (np.abs(self.value - old_value) > self.tolerance).any():
            old_value = np.copy(self.value)
            for state in np.ndindex(*self.mdp.observation_space.nvec):
                if state in self.mdp.terminals:
                    continue

                expected_returns = []
                for action in np.arange(self.mdp.action_space.n):
                    expected_returns.append(
                        self.calculate_expected_return(state, action)
                    )
                self.value[state] = np.max(expected_returns)

            self.update_policy()
        return self.value, self.policy
