"""
grid_world.py
GridWorld environment
"""

import random
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

random.seed(42)
Coord = Tuple[int, int]
Block = List[Coord]


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

        # Calculate the next state based on the action
        if action == 0:  # Right
            next_state = (self.state[0], self.state[1] + 1)
        elif action == 1:  # Down
            next_state = (self.state[0] + 1, self.state[1])
        elif action == 2:  # Left
            next_state = (self.state[0], self.state[1] - 1)
        elif action == 3:  # Up
            next_state = (self.state[0] - 1, self.state[1])
        else:
            raise ValueError("Invalid action")

        # Check if the next state is valid
        if next_state[0] < 0 or next_state[0] >= self.size[0]:
            next_state = self.state
        if next_state[1] < 0 or next_state[1] >= self.size[1]:
            next_state = self.state
        for obstacle in self.obstacles:
            if next_state in obstacle:
                next_state = self.state
                break
        self.state = next_state
        terminated = self.state in self.goal

        # Calculate the reward
        if terminated:
            reward = self.rewards["goal"]
        elif self.state in self.obstacles:
            reward = self.rewards["obstacle"]
        else:
            reward = self.rewards["default"]

        return self.state, reward, terminated, False, {}

    def reset(self, seed: int = None) -> Tuple[Coord, Dict]:
        super().reset(seed=seed)
        self.state = self.start
        return self.state, {}

    def render(self) -> str:
        grid = np.full(self.size, ".")
        grid[self.start] = "S"
        for goal in self.goal:
            grid[goal] = "G"
        for obstacle in self.obstacles:
            for coord in obstacle:
                grid[coord] = "X"
        return "\n".join([" ".join(row) for row in grid])
