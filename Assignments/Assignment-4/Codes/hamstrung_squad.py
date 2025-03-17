"""
hamstrung_squad.py
Hamstrung squad game environment
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

Coord = Tuple[int, int]
Coords = List[Coord]
ObsType = NDArray
ActType = Tuple[int, int]


class HamstrungSquadEnv(gym.Env):
    """Hamstrung squad game environment"""

    metadata = {"render.modes": ["ansi", "rgb_array"], "render.fps": 4}

    def __init__(
        self,
        grid_size: int = 10,
        max_payoff: int = 10,
        rewards: Dict[str, int] = {"capture": 10, "default": -1},
        render_mode: str = "ansi",
        seed: int = 42,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.max_payoff = max_payoff
        self.rewards = rewards
        self.render_mode = render_mode
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(2), gym.spaces.Discrete(4))
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=grid_size - 1, shape=(5,), dtype=np.int32
        )
        self.payoff_table = np.full((grid_size, grid_size), np.nan)
        self.seed(seed=seed)

    def seed(self, seed: int = None) -> None:
        self._seed = seed
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def _get_obs(self) -> ObsType:
        return np.array([*self.pursuer, *self.evader, self.pursuer_direction])

    def reset(
        self,
        evader: Coord,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict]:
        super().reset(seed=seed)
        self.pursuer: Coord = np.array([self.grid_size - 1, 0])
        self.pursuer_direction: int = 0  # 0: Up, 1: Right, 2: Down, 3: Left
        self.evader: Coord = np.array(evader)
        self.payoff: int = 0
        info = {
            "pursuer_start": self.pursuer,
            "evader_start": self.evader,
            "pursuer_direction": self.pursuer_direction,
            "seed": self._seed,
        }
        return self._get_obs(), info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
        pursuer_action, evader_action = action

        # Pursuer moves
        if pursuer_action == 1:  # Right
            self.pursuer_direction = (self.pursuer_direction + 1) % 4
        pursuer_delta = np.array(
            [(-2, 0), (0, 2), (2, 0), (0, -2)][self.pursuer_direction]
        )
        self.pursuer = np.clip(self.pursuer + pursuer_delta, 0, self.grid_size - 1)

        # Evader moves
        evader_delta = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)][evader_action])
        self.evader = np.clip(self.evader + evader_delta, 0, self.grid_size - 1)

        # Check termination
        self.payoff += 1
        terminated: bool = np.linalg.norm(self.pursuer - self.evader) <= 1.5
        truncated: bool = self.payoff >= self.max_payoff
        reward = self.rewards["capture"] if terminated else self.rewards["default"]
        if terminated:
            self.payoff_table[tuple(self.evader)] = self.payoff
        elif truncated:
            self.payoff_table[tuple(self.evader)] = -1
        return self._get_obs(), reward, terminated, truncated, {}

    def _show_payoff_table(self) -> None:
        print("Payoff table:")
        show = lambda x: f"{x:2.0f}" if not (np.isnan(x) or x == -1) else " ."
        for row in self.payoff_table:
            print(" ".join(show(x) for x in row))

    def render(self) -> None:
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        else:
            raise NotImplementedError

    def _create_grid(self) -> NDArray:
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        grid[self.evader[0], self.evader[1]] = "E"
        grid[self.pursuer[0], self.pursuer[1]] = "P"
        return grid

    def _render_ansi(self, use_color: bool = True) -> None:
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

    def _render_rgb_array(self) -> None:
        color_map = {
            ".": np.array([255, 255, 255]),  # White
            "P": np.array([0, 0, 255]),  # Blue
            "E": np.array([255, 0, 0]),  # Red
        }
        grid = self._create_grid()
        rgb_array = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                rgb_array[i, j] = color_map[cell]
        plt.imshow(rgb_array)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
