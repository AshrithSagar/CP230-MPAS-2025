"""
hamstrung_squad.py
Hamstrung sqaud game environment
"""

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

Coord = Tuple[int, int]
Coords = List[Coord]


class HamstrungSquadEnv(gym.Env):
    """Hamstrung squad game environment"""

    metadata = {"render.modes": ["ansi", "rgb_array"], "render.fps": 4}

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 10,
        render_mode: str = "ansi",
        seed: int = 42,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(2), gym.spaces.Discrete(4))
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=self.grid_size - 1, shape=(5,), dtype=np.int32
        )
        self.seed(seed=seed)

    class Action(IntEnum):
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

    def seed(self, seed: int = None) -> None:
        self._seed = seed
        self.reset(seed=seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def _get_obs(self):
        return np.array(
            [
                self.pursuer[0],
                self.pursuer[1],
                self.evader[0],
                self.evader[1],
                self.pursuer_direction,
            ],
            dtype=np.int32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray, Dict]:
        super().reset(seed=seed)
        self.pursuer: Coord = np.array([1, self.grid_size - 2])
        self.pursuer_direction = self.Action.UP
        self.evader: Coord = np.array([self.grid_size - 1, 0])
        return self._get_obs(), {}

    def _direction_to_delta(self, direction):
        return np.array([(0, -1), (1, 0), (0, 1), (-1, 0)][direction])

    def step(self, action: int) -> Tuple[NDArray, float, bool, bool, Dict]:
        pursuer_action, evader_action = action

        # Pursuer moves
        if pursuer_action == 1:
            self.pursuer_direction = (self.pursuer_direction + 1) % 4
        delta = self._direction_to_delta(self.pursuer_direction)
        self.pursuer = np.clip(self.pursuer + delta * 2, 0, self.grid_size - 1)

        # Evader moves
        evader_delta = [(-1, 0), (1, 0), (0, -1), (0, 1)][evader_action]
        self.evader = np.clip(self.evader + evader_delta, 0, self.grid_size - 1)

        # Check termination
        done = np.linalg.norm(self.pursuer - self.evader) <= 1.5
        reward = -10 if done else -1
        return self._get_obs(), reward, done, False, {}

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        else:
            raise NotImplementedError

    def _create_grid(self) -> NDArray:
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        grid[self.pursuer[1], self.pursuer[0]] = "P"
        grid[self.evader[1], self.evader[0]] = "E"
        return grid

    def _render_ansi(self, use_color: bool = True) -> str:
        if use_color:
            color_map = {
                ".": "\033[0m",  # Default
                "P": "\033[94m",  # Blue
                "E": "\033[91m",  # Red
            }
        else:
            color_map = {".": "", "P": "", "E": ""}
        grid = self._create_grid()
        ansi = "\n".join(" ".join(f"{color_map[c]}{c}" for c in row) for row in grid)
        print(ansi, end="\033[0m\n" if use_color else "\n")
        return ansi

    def _render_rgb_array(self) -> NDArray:
        color_map = {
            ".": np.array([255, 255, 255]),  # White
            "P": np.array([0, 0, 255]),  # Blue
            "E": np.array([255, 0, 0]),  # Red
        }
        grid = self._create_grid()
        rgb_array = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                rgb_array[i, j] = color_map[cell]
        plt.imshow(rgb_array)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
        return rgb_array
