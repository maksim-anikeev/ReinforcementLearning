import gym
import numpy as np
import matplotlib.pyplot as plt

from gamblers import Gambler

from typing import List, Dict, Tuple


class ArmedBanditTestbedEnv(gym.Env):
    def __init__(self, n: int = 10, stationary: bool = True):
        self.action_space = gym.spaces.Discrete(n)
        self.stationary = stationary

        self.np_random = None
        self.seed()

        self.q_star = None
        self.reset()

    def step(self, arm: int) -> Tuple[int, float, bool, Dict[str, int]]:
        assert self.action_space.contains(arm)

        if not self.stationary:
            q_drift = self.np_random.normal(loc=0.0, scale=0.01, size=self.action_space.n)
            self.q_star += q_drift
        reward = self.np_random.normal(loc=self.q_star[arm], scale=1.0)
        return 0, reward, False, {'arm_star': np.argmax(self.q_star)}

    def reset(self):
        if self.stationary:
            self.q_star = self.np_random.normal(loc=0., scale=1., size=self.action_space.n)
        else:
            self.q_star = np.zeros(shape=[self.action_space.n])

    def render(self, mode: str = 'human'):
        bandit_samples = []
        for arm in range(self.action_space.n):
            bandit_samples += [np.random.normal(loc=self.q_star[arm], scale=1., size=1000)]
        plt.violinplot(bandit_samples, showmeans=True)
        plt.xlabel('Bandit Arm')
        plt.ylabel('Reward Distribution')
        plt.show()

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

gym.envs.registration.register(
    id='ArmedBanditTestbed-v0',
    entry_point=lambda n, stationary: ArmedBanditTestbedEnv(n, stationary),
    max_episode_steps = 10000,
    nondeterministic=True,
    reward_threshold=1.0,
    kwargs={'n': 10, 'stationary': True}
)

def run_episode(bandit, gambler: Gambler, seed: int = None) -> Tuple[List[float], List[int], int]:
    rewards = []
    corrects = []
    steps = 0

    bandit.seed(seed)
    bandit.reset()
    gambler.seed(seed)
    gambler.reset()

    done = False
    while not done:
        arm = gambler.arm()
        state, reward, done, info = bandit.step(arm)
        gambler.update(arm, reward)

        rewards.append(reward)
        corrects.append(1 if arm == info.get('arm_star') else 0)
        steps += 1

    rewards = np.array(rewards, dtype=float)
    corrects = np.array(corrects, dtype=int)
    return rewards, corrects, steps

