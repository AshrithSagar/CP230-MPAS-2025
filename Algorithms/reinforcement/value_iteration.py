"""
value_iteration.py
Value Iteration algorithm
"""

import timeit
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from rich.progress import Progress
from utils.grid_world import Coord, GridWorld, Path


class ValueIterationAgent:
    """Value Iteration algorithm"""

    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.9,
        initial_v_table: np.ndarray = None,
    ):
        self.env = env
        self.gamma = gamma

        if initial_v_table is not None:
            assert initial_v_table.shape == env.observation_space.nvec
            self.v_table = initial_v_table
        else:
            self.v_table = np.zeros(env.observation_space.nvec)

    def _get_v_value(self, state: Coord) -> np.ndarray:
        return self.v_table[state[0], state[1]]

    def _set_v_value(self, state: Coord, value: float) -> None:
        self.v_table[state[0], state[1]] = value

    def _get_q_values(self, state: Coord) -> NDArray:
        q_values = np.zeros(self.env.action_space.n)
        for action in range(self.env.action_space.n):
            next_state = self.env.unwrapped._get_next_state(state, action)
            reward = self.env.unwrapped._get_reward(next_state)
            q_value = reward + self.gamma * self._get_v_value(next_state)
            q_values[action] = q_value
        return q_values

    def _train_episode(self) -> None:
        """Train the agent for a single episode"""
        for state in np.ndindex(self.v_table.shape):
            if self.env.unwrapped._in_terminal(state):
                continue
            q_values = self._get_q_values(state)
            self._set_v_value(state, np.max(q_values))

    def train(
        self,
        episodes: int = None,
        threshold: float = 1e-4,
        timed: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {"threshold": threshold}
        episode, converged, prev_v_table = 1, False, np.copy(self.v_table)
        start_time = timeit.default_timer() if timed else None
        with Progress(transient=True) as progress:
            task = progress.add_task("[green]Training...", total=episodes)
            while True:
                self._train_episode()
                progress.advance(task)
                if episodes and episode >= episodes:
                    break
                if np.allclose(self.v_table, prev_v_table, atol=threshold):
                    converged = True
                    break
                prev_v_table = np.copy(self.v_table)
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
            action = np.argmax(self._get_q_values(state))
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
