import scipy
import numpy as np
import gymnasium as gym
from typing import Tuple, List

from mdp.mdp import MDP


class JacksCarRental(MDP):
    def __init__(
        self,
        max_transfers: int = 5,
        max_cars: int = 20,
        transfer_cost: int = 2,
        rental_price: int = 10,
        requests_lambda: Tuple[int, int] = (3, 4),
        returns_lambda: Tuple[int, int] = (3, 2),
        gamma: float = 0.9,
    ) -> None:
        super().__init__()
        self.max_transfers = max_transfers
        self.max_cars = max_cars
        self.transfer_cost = transfer_cost
        self.rental_price = rental_price
        self.requests_lambda = requests_lambda
        self.returns_lambda = returns_lambda
        self.gamma = gamma

        self.action_space = gym.spaces.Discrete(
            2 * max_transfers + 1, start=-max_transfers
        )
        self.observation_space = gym.spaces.MultiDiscrete([max_cars + 1, max_cars + 1])

        self._init_dist()
        self.reset()

    def _init_dist(self) -> None:
        n_requests_returns = np.indices([15] * 4).reshape([4, -1]).T
        self.requests = n_requests_returns[:, :2]
        requests_dist = scipy.stats.poisson(self.requests_lambda)
        requests_probs = requests_dist.pmf(self.requests)
        self.returns = n_requests_returns[:, 2:]
        returns_dist = scipy.stats.poisson(self.returns_lambda)
        returns_probs = returns_dist.pmf(self.returns)
        self.probs = np.prod(requests_probs * returns_probs, axis=1)
        self.probs /= np.sum(self.probs)

    def reset(self) -> None:
        self.state = np.random.randint(0, self.max_cars, 2)

    def transitions(
        self, state: List[int], action: int
    ) -> Tuple[List[List[int]], List[int], List[float], List[bool]]:
        state, costs = self.transfer(state, action)
        transitions, rented = self.rent_cars(state)
        rewards = self.calculate_profit(transitions, costs, rented)
        transitions = self.return_cars(transitions)
        terminals = np.zeros_like(rewards, dtype=bool)
        return transitions, rewards, self.probs, terminals

    def transfer(self, state: List[int], action: int) -> Tuple[Tuple[int, int], int]:
        costs = self.transfer_cost * np.abs(action)
        if action > 0:
            transferred = np.min([np.abs(action), state[0], self.max_cars - state[1]])
        else:
            transferred = -np.min([np.abs(action), state[1], self.max_cars - state[0]])
        state += np.array([-transferred, transferred])
        return state, costs

    def rent_cars(self, state: List[int]) -> Tuple[List[List[int]], List[List[int]]]:
        rented = np.minimum(self.requests, state)
        transitions = state - rented
        return transitions, rented

    def return_cars(self, transitions: List[List[int]]) -> List[List[int]]:
        returned = np.minimum(self.returns, self.max_cars - transitions)
        return transitions + returned

    def calculate_profit(
        self, transitions: List[List[int]], costs: int, rented: List[List[int]]
    ) -> List[int]:
        revenue = self.rental_price * np.sum(rented, axis=1)
        return revenue - costs
