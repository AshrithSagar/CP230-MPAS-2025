"""
grid_world.py
GridWorld environment
"""

from enum import IntEnum
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cityblock

Coord = Tuple[int, int]
Block = List[Coord]
Path = List[Coord]


class GridWorld(gym.Env):
    """GridWorld environment"""

    metadata = {"render.modes": ["ansi", "rgb_array"]}

    def __init__(
        self,
        size: Tuple[int, int],
        start: Coord,
        goal: Block,
        obstacles: List[Block],
        rewards: Dict[str, int] = {"goal": 100, "obstacle": -10, "default": -1},
        slippage: float = None,
        obstacle_penalty: float = None,
        seed: int = 42,
    ):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.rewards = rewards
        self.slippage = slippage
        self.obstacle_penalty = obstacle_penalty
        self.action_space = gym.spaces.Discrete(4, seed=seed)
        self.observation_space = gym.spaces.Discrete(size[0] * size[1], seed=seed)
        self.reset(seed=seed)
        self._seed = seed

    class Action(IntEnum):
        RIGHT = 0
        DOWN = 1
        LEFT = 2
        UP = 3

    def _get_obstacle_penalty(self, state: Coord) -> int:
        """Get the penalty for being close to an obstacle"""
        if not self.obstacle_penalty:
            return 0
        min_distance = min(cityblock(state, c) for obs in self.obstacles for c in obs)
        return round(self.obstacle_penalty * min_distance)

    def step(self, action: int) -> Tuple[Coord, int, bool, bool, Dict]:
        if self.slippage and np.random.rand() < self.slippage:
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
            action = np.random.choice(perpendicular_actions[action])
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
                else self.rewards["default"] - self._get_obstacle_penalty(self.state)
            )
        )
        return self.state, reward, terminated, False, {}

    def reset(self, seed: int = None) -> Tuple[Coord, Dict]:
        super().reset(seed=seed)
        self.state = self.start
        return self.state, {}

    def render(self, mode: str = "ansi", path: Path = None, show: bool = True):
        grid = np.full(self.size, ".")
        for obstacle in self.obstacles:
            grid[tuple(zip(*obstacle))] = "X"
        if path:
            grid[tuple(zip(*path))] = "*"
        grid[self.start] = "S"
        grid[tuple(zip(*self.goal))] = "G"
        if mode == "ansi":
            ansi = "\n".join([" ".join(row) for row in grid])
            if show:
                print(ansi)
            return ansi
        elif mode == "rgb_array":
            color_map = {
                ".": np.array([255, 255, 255]),
                "*": np.array([255, 255, 0]),
                "S": np.array([0, 0, 255]),
                "G": np.array([0, 255, 0]),
                "X": np.array([255, 0, 0]),
            }
            rgb_array = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    cell = grid[i, j]
                    rgb_array[i, j] = color_map[cell]
            if show:
                plt.imshow(rgb_array)
                plt.show()
            return rgb_array
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
