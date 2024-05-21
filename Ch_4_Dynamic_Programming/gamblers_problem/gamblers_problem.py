import numpy as np
import gymnasium as gym

from mdp.mdp import MDP


class GamblersProblem(MDP):
    def __init__(
        self,
        p_win: float = 0.5,
        min_bet: int = 1,
        max_capital: int = 100,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.p_win = p_win
        self.min_bet = min_bet
        self.max_capital = max_capital
        self.gamma = gamma

        self.action_space = gym.spaces.Discrete(max_capital - min_bet)
        self.observation_space = gym.spaces.MultiDiscrete([max_capital + 1])
        self.terminals = [np.array([0]), np.array([max_capital])]

        self.reset()

    def reset(self) -> int:
        self.state = self.observation_space.sample()
        return self.state

    def transitions(self, capital: int, bet: int) -> None:
        transitions = np.zeros((2, 1), dtype=int)
        rewards = np.zeros(2, dtype=float)
        probs = np.zeros(2, dtype=float)
        terminals = np.zeros(2, dtype=bool)

        bet = min(capital, max(bet, self.min_bet))
        capital_loss = max(capital - bet, self.terminals[0])
        transitions[0, 0] = capital_loss
        rewards[0] = 0.0
        probs[0] = 1 - self.p_win
        terminals[0] = capital_loss == self.terminals[0]

        capital_win = min(capital + bet, self.terminals[1])
        transitions[1, 0] = capital_win
        rewards[1] = capital_win == self.terminals[1]
        probs[1] = self.p_win
        terminals[1] = capital_win == self.terminals[1]

        return transitions, rewards, probs, terminals
