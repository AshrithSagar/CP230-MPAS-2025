"""
grid_world.py
GridWorld environment
"""

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cityblock

Coord = Tuple[int, int]
Block = List[Coord]
Path = List[Coord]


class GridWorld(gym.Env):
    """GridWorld environment"""

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        size: Tuple[int, int],
        start: Coord,
        goal: Block,
        obstacles: List[Block],
        rewards: Dict[str, int] = {"goal": 100, "obstacle": -10, "default": -1},
        slippage: float = None,
        obstacle_penalty: float = None,
        render_mode: str = "ansi",
        seed: int = 42,
    ):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.rewards = rewards
        self.slippage = slippage
        self.obstacle_penalty = obstacle_penalty
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.MultiDiscrete([size[0], size[1]])
        self.seed(seed=seed)
        self.path: Path = None
        self.obstacle_penalties = self._compute_obstacle_penalties()

    class Action(IntEnum):
        RIGHT = 0
        DOWN = 1
        LEFT = 2
        UP = 3

    def seed(self, seed: int = None) -> List[int]:
        self._seed = seed
        self.reset(seed=seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def _compute_obstacle_penalties(self) -> np.ndarray:
        """Compute the penalty for being close to an obstacle."""
        penalties = np.zeros(self.size)
        if self.obstacle_penalty is None:
            return penalties
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                min_distance = float("inf")
                for obstacle in self.obstacles:
                    for coord in obstacle:
                        distance = cityblock((i, j), coord)
                        if distance != 0:
                            min_distance = min(min_distance, 1 / distance)
                penalties[i, j] = min_distance if min_distance != float("inf") else 0
        penalties = np.round(penalties * self.obstacle_penalty).astype(int)
        return penalties

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.slippage and self.np_random.random() < self.slippage:
            perpendicular_actions = {
                GridWorld.Action.UP.value: [
                    GridWorld.Action.RIGHT.value,
                    GridWorld.Action.LEFT.value,
                ],
                GridWorld.Action.DOWN.value: [
                    GridWorld.Action.RIGHT.value,
                    GridWorld.Action.LEFT.value,
                ],
                GridWorld.Action.RIGHT.value: [
                    GridWorld.Action.UP.value,
                    GridWorld.Action.DOWN.value,
                ],
                GridWorld.Action.LEFT.value: [
                    GridWorld.Action.UP.value,
                    GridWorld.Action.DOWN.value,
                ],
            }
            action = self.np_random.choice(perpendicular_actions[action])
        next_state = {
            GridWorld.Action.RIGHT.value: (self.state[0], self.state[1] + 1),
            GridWorld.Action.DOWN.value: (self.state[0] + 1, self.state[1]),
            GridWorld.Action.LEFT.value: (self.state[0], self.state[1] - 1),
            GridWorld.Action.UP.value: (self.state[0] - 1, self.state[1]),
        }.get(action, self.state)
        if (
            not (0 <= next_state[0] < self.size[0])
            or not (0 <= next_state[1] < self.size[1])
            or any(next_state in obstacle for obstacle in self.obstacles)
        ):
            next_state = self.state
        self.state = next_state
        terminated = self.state in self.goal
        reward = (
            self.rewards["goal"]
            if terminated
            else (
                self.rewards["obstacle"]
                if any(self.state in obstacle for obstacle in self.obstacles)
                else self.rewards["default"] - self.obstacle_penalties[self.state]
            )
        )
        return np.array(self.state), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.state = self.start
        return np.array(self.state), {}

    def render(self):
        grid = self._create_grid()
        render_methods = {
            "ansi": self._render_ansi,
            "rgb_array": self._render_rgb_array,
        }
        if mode := render_methods.get(self.render_mode):
            return mode(grid)
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def _create_grid(self) -> np.ndarray:
        grid = np.full(self.size, ".", dtype=str)
        for obstacle in self.obstacles:
            grid[tuple(zip(*obstacle))] = "X"
        if self.path:
            grid[tuple(zip(*self.path))] = "*"
        grid[self.start] = "S"
        grid[tuple(zip(*self.goal))] = "G"
        return grid

    def _render_ansi(self, grid: np.ndarray, use_color: bool = True) -> str:
        if use_color:
            color_map = {
                ".": "\033[0m",  # Default
                "*": "\033[33m",  # Yellow
                "S": "\033[94m",  # Blue
                "G": "\033[92m",  # Green
                "X": "\033[91m",  # Red
            }
        else:
            color_map = {".": "", "*": "", "S": "", "G": "", "X": ""}
        ansi = "\n".join(" ".join(f"{color_map[c]}{c}" for c in r) for r in grid)
        print(ansi, end="\033[0m\n" if use_color else "\n")
        return ansi

    def _render_rgb_array(self, grid: np.ndarray) -> np.ndarray:
        color_map = {
            ".": np.array([255, 255, 255]),  # White
            "*": np.array([255, 255, 0]),  # Yellow
            "S": np.array([0, 0, 255]),  # Blue
            "G": np.array([0, 255, 0]),  # Green
            "X": np.array([255, 0, 0]),  # Red
        }
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
