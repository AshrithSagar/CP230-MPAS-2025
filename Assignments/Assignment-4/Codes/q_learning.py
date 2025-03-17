"""
q_learning.py
Q-Learning algorithm
"""

import timeit
from typing import Any, Dict, Optional

import numpy as np
from hamstrung_squad import ActType, Coord, HamstrungSquadEnv, ObsType
from numpy.typing import NDArray
from tqdm import tqdm


class QLearningAgent:
    """Q-Learning algorithm"""

    def __init__(
        self,
        env: HamstrungSquadEnv,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        initial_q_table: np.ndarray = None,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        grid_size = env.grid_size
        q_table_shape = (grid_size, grid_size, grid_size, grid_size, 4, 2, 4)
        if initial_q_table is not None:
            assert initial_q_table.shape == q_table_shape, "Invalid Q-table shape"
            self.q_table = initial_q_table
        else:
            self.q_table = np.zeros(q_table_shape)

    def _get_q_value(self, obs: ObsType, action: Optional[ActType] = None) -> NDArray:
        px, py, ex, ey, pd = obs
        if action is not None:
            pa, ea = action
            return self.q_table[px, py, ex, ey, pd, pa, ea]
        else:
            return self.q_table[px, py, ex, ey, pd]

    def _set_q_value(self, obs: ObsType, action: ActType, value: float) -> None:
        px, py, ex, ey, pd = obs
        pa, ea = action
        self.q_table[px, py, ex, ey, pd, pa, ea] = value

    def _train_episode(self, evader: Coord, epsilon: float = None) -> None:
        """Train the agent for a single episode"""
        obs, _ = self.env.reset(evader)
        epsilon = epsilon or self.epsilon
        terminated, truncated = False, False
        while not (terminated or truncated):
            if self.env.np_random.random() < epsilon:
                action: ActType = self.env.action_space.sample()
            else:
                action: ActType = np.argmax(self._get_q_value(obs))
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            q_value = self._get_q_value(obs, action) + self.alpha * (
                reward
                + self.gamma * np.max(self._get_q_value(next_obs))
                - self._get_q_value(obs, action)
            )
            self._set_q_value(obs, action, q_value)
            obs = next_obs

    def _train_evader(
        self,
        evader: Coord,
        episodes: int = None,
        threshold: float = 1e-4,
        decay_epsilon: callable = None,
    ) -> None:
        """Train the agent for a given evader position"""
        episode, epsilon = 1, self.epsilon
        prev_q_table = np.copy(self.q_table[:, :, evader[0], evader[1], :, :, :])
        while True:
            self._train_episode(evader=evader, epsilon=epsilon)
            if decay_epsilon:
                epsilon = decay_epsilon(epsilon)
            if (episodes and episode >= episodes) or np.allclose(
                self.q_table[:, :, evader[0], evader[1], :, :, :],
                prev_q_table,
                atol=threshold,
            ):
                break
            prev_q_table = self.q_table[:, :, evader[0], evader[1], :, :, :]
            episode += 1

    def train(
        self,
        episodes: int = None,
        threshold: float = 1e-4,
        decay_epsilon: callable = None,
        timed: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train the agent"""
        info: Dict[str, Any] = {"threshold": threshold}
        start_time = timeit.default_timer() if timed else None
        pbar = tqdm(desc="Training", total=self.env.grid_size**2, leave=False)
        for evader in np.ndindex(self.env.grid_size, self.env.grid_size):
            self._train_evader(
                evader=(self.env.grid_size - evader[0], evader[1]),
                episodes=episodes,
                threshold=threshold,
                decay_epsilon=decay_epsilon,
            )
            pbar.update()
        pbar.close()
        if timed:
            elapsed = timeit.default_timer() - start_time
            info["time"] = elapsed
        if verbose:
            print("Training completed:")
            if timed:
                print(f" Time: {elapsed:.3f}s")
        return info
