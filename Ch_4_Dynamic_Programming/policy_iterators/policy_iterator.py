import numpy as np
from typing import Tuple

from mdp.mdp import MDP


class PolicyIterator:
    def __init__(self, mdp: MDP) -> None:
        self.mdp = mdp
        self.value = np.zeros(shape=mdp.observation_space.nvec)
        self.policy = np.random.randint(
            mdp.action_space.start,
            mdp.action_space.start + mdp.action_space.n - 1,
            size=self.value.shape,
        )

    def find_optimal_policy(self) -> None:
        old_policy = self.policy - np.inf
        while (self.policy != old_policy).any():
            old_policy = np.copy(self.policy)
            for state in np.ndindex(*self.mdp.observation_space.nvec):
                if state in self.mdp.terminals:
                    self.policy[state] = 0
                    continue

                action = self.policy[state]
                self.value[state] = self.calculate_expected_return(state, action)
            self.update_policy()
        return self.value, self.policy

    def update_policy(self) -> None:
        for state in np.ndindex(*self.mdp.observation_space.nvec):
            if state in self.mdp.terminals:
                self.policy[state] = 0
                continue

            expected_returns = []
            for action in np.arange(
                self.mdp.action_space.start,
                self.mdp.action_space.start + self.mdp.action_space.n,
            ):
                G = self.calculate_expected_return(state, action)
                expected_returns.append(G)

            self.policy[state] = (
                np.argmax(expected_returns) + self.mdp.action_space.start
            )

    def calculate_expected_return(self, state: Tuple[int], action: int) -> float:
        transitions, rewards, probs, terminals = self.mdp.transitions(
            np.array(state), action
        )
        returns = rewards + self.mdp.gamma * self.value[tuple(transitions.T)]
        terminal = np.where(terminals)[0]
        if terminal.size > 0:
            self.value[tuple(transitions[terminal].T)] = rewards[terminal]
            returns[terminal] = rewards[terminal]
        return np.sum(probs * returns)
