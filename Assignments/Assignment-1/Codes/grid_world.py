"""
grid_world.py
GridWorld environment
"""

from enum import IntEnum
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

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
        rewards: Dict[str, int] = {"goal": 1, "obstacle": -1, "default": 0},
    ):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.rewards = rewards
        self.state = start
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(size[0] * size[1])

    class Action(IntEnum):
        RIGHT = 0
        DOWN = 1
        LEFT = 2
        UP = 3

    def step(self, action: int) -> Tuple[Coord, int, bool, bool, Dict]:
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
                else self.rewards["default"]
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
