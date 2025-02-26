"""
grid_world.py
GridWorld environment
"""

from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

Coord = Tuple[int, int]
Block = List[Coord]
Path = List[Coord]


class GridWorld(gym.Env):
    """GridWorld environment"""

    metadata = {"render.modes": ["ansi"]}

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

    def step(self, action: int) -> Tuple[Coord, int, bool, bool, Dict]:
        next_state = {
            0: (self.state[0], self.state[1] + 1),  # Right
            1: (self.state[0] + 1, self.state[1]),  # Down
            2: (self.state[0], self.state[1] - 1),  # Left
            3: (self.state[0] - 1, self.state[1]),  # Up
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

    def render(self, path: Path = None) -> str:
        grid = np.full(self.size, ".")
        if path is not None:
            for coord in path:
                grid[coord] = "*"
        grid[self.start] = "S"
        for goal in self.goal:
            grid[goal] = "G"
        for obstacle in self.obstacles:
            for coord in obstacle:
                grid[coord] = "X"
        return "\n".join([" ".join(row) for row in grid])
