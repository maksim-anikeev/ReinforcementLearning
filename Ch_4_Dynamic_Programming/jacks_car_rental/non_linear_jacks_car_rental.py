import numpy as np
from typing import List, Tuple

from .jacks_car_rental import JacksCarRental


class NonLinearJacksCarRental(JacksCarRental):
    def __init__(
        self,
        max_transfers: int = 5,
        max_cars: int = 20,
        transfer_cost: int = 2,
        rental_price: int = 10,
        max_free_parking: int = 10,
        parking_price: int = 4,
        requests_lambda: Tuple[int, int] = (3, 4),
        returns_lambda: Tuple[int, int] = (3, 2),
        gamma: float = 0.9,
    ) -> None:
        super().__init__(
            max_transfers,
            max_cars,
            transfer_cost,
            rental_price,
            requests_lambda,
            returns_lambda,
            gamma,
        )
        self.max_free_parking = max_free_parking
        self.parking_price = parking_price

    def transfer(self, state: List[int], action: int) -> Tuple[Tuple[int] | int]:
        state, costs = super().transfer(state, action)
        if action > 0:
            costs -= self.transfer_cost
        return state, costs

    def calculate_profit(
        self, transitions: List[List[int]], costs: int, rented: List[List[int]]
    ) -> List[int]:
        profit = super().calculate_profit(transitions, costs, rented)
        parking_costs = self.parking_price * np.sum(
            (self.max_free_parking < transitions), axis=1
        )
        return profit - parking_costs
