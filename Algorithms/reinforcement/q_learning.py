"""
q_learning.py
Q-Learning algorithm
"""

import timeit
from typing import Any, Dict, Tuple

import numpy as np
from rich.progress import Progress
from utils.grid_world import Coord, GridWorld, Path


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
                *env.observation_space.nvec,
                env.action_space.n,
            )
            self.q_table = initial_q_table
        else:
            self.q_table = np.zeros((*env.observation_space.nvec, env.action_space.n))

    def _get_q_value(self, state: Coord, action: int = None) -> np.ndarray:
        if action is not None:
            return self.q_table[state[0], state[1], action]
        return self.q_table[state[0], state[1]]

    def _set_q_value(self, state: Coord, action: int, value: float) -> None:
        self.q_table[state[0], state[1], action] = value

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

    def train(
        self,
        episodes: int = None,
        threshold: float = 1e-4,
        decay_epsilon: callable = None,
        timed: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {"threshold": threshold}
        episode, converged = 1, False
        prev_q_table, epsilon = np.copy(self.q_table), self.epsilon
        start_time = timeit.default_timer() if timed else None
        with Progress(transient=True) as progress:
            task = progress.add_task("[green]Training...", total=episodes)
            while True:
                self._train_episode(epsilon=epsilon)
                if decay_epsilon:
                    epsilon = decay_epsilon(epsilon)
                progress.advance(task)
                if episodes and episode >= episodes:
                    break
                if np.allclose(self.q_table, prev_q_table, atol=threshold):
                    converged = True
                    break
                prev_q_table = np.copy(self.q_table)
                episode += 1
        if timed:
            elapsed = timeit.default_timer() - start_time
            info["time"] = elapsed
        info["episodes"] = episode
        if verbose:
            print("Training completed:")
            print(f" Episodes: {episode}")
            if converged:
                print(f" Threshold: {threshold}")
            if timed:
                print(f" Time: {elapsed:.3f}s")
        return info

    def test(self, max_steps: int = None, verbose: bool = True) -> Tuple[Path, float]:
        if max_steps is None:
            max_steps = self.env.unwrapped.size[0] * self.env.unwrapped.size[1]
        state, _ = self.env.reset()
        path, done, steps, total_reward = [state], False, 0, 0.0
        while not done and steps < max_steps:
            action = np.argmax(self._get_q_value(state))
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            path.append(next_state)
            total_reward += reward
            state = next_state
            steps += 1
        self.env.unwrapped.path = path
        if verbose:
            print(f"Total reward: {total_reward}")
            print(f"Path length: {len(path)}")
        return path, total_reward
