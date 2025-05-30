"""
q_learning.py
Q-Learning algorithm
"""

from typing import List, Tuple

import numpy as np
from grid_world import Coord, GridWorld
from rich.progress import Progress


class QLearningAgent:
    """Q-Learning algorithm"""

    def __init__(
        self,
        env: GridWorld,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        initial_q_table: np.ndarray = None,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        if initial_q_table is not None:
            assert initial_q_table.shape == (
                env.observation_space.n,
                env.action_space.n,
            )
            self.q_table = initial_q_table
        else:
            self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def _get_q_value(self, state: Coord, action: int = None) -> np.ndarray:
        index = state[0] * self.env.size[1] + state[1]
        if action is not None:
            return self.q_table[index, action]
        return self.q_table[index]

    def _set_q_value(self, state: Coord, action: int, value: float) -> None:
        index = state[0] * self.env.size[1] + state[1]
        self.q_table[index, action] = value

    def _train_episode(self, epsilon: float = None) -> None:
        """Train the agent for a single episode"""
        state, _ = self.env.reset()
        epsilon = epsilon or self.epsilon
        terminated = False
        while not terminated:
            if np.random.rand() < epsilon:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self._get_q_value(state))
            next_state, reward, terminated, _, _ = self.env.step(action)
            q_value = self._get_q_value(state, action) + self.alpha * (
                reward
                + self.gamma * np.max(self._get_q_value(next_state))
                - self._get_q_value(state, action)
            )
            self._set_q_value(state, action, q_value)
            state = next_state

    def train(self, episodes: int, decay_epsilon: callable = None) -> None:
        with Progress() as progress:
            task = progress.add_task("[green]Training...", total=episodes)
            epsilon = self.epsilon
            for _ in range(episodes):
                self._train_episode(epsilon=epsilon)
                if decay_epsilon is not None:
                    epsilon = decay_epsilon(epsilon)
                progress.advance(task)

    def test(self, max_steps: int = None) -> Tuple[List[Coord], int]:
        if max_steps is None:
            max_steps = self.env.size[0] * self.env.size[1]
        state, _ = self.env.reset()
        path = [state]
        total_reward = 0
        terminated = False
        steps = 0
        while not terminated and steps < max_steps:
            action = np.argmax(self._get_q_value(state))
            next_state, reward, terminated, _, _ = self.env.step(action)
            path.append(next_state)
            total_reward += reward
            state = next_state
            steps += 1
        return path, total_reward
