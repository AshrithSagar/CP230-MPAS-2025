"""
hamstrung_squad.py
Hamstrung sqaud game environment
"""

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

Coord = Tuple[int, int]
Coords = List[Coord]


class HamstrungSquadEnv(gym.Env):
    """Hamstrung squad game environment"""

    metadata = {"render.modes": ["ansi"]}

    def __init__(
        self,
        size: Tuple[int, int],
        start: Dict[str, Union[Coord, Coords]],
        max_steps: int = 10,
        render_mode: str = "ansi",
        seed: int = 42,
    ):
        super().__init__()
        self.size = size
        assert "pursuer" in start and "evader" in start
        self.start = start
        self.pursuer = start["pursuer"]
        self.evader = start["evader"]
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.MultiDiscrete([*size, *size])
        self.seed(seed=seed)

    class Action(IntEnum):
        RIGHT = 0
        DOWN = 1
        LEFT = 2
        UP = 3

    def seed(self, seed: int = None) -> None:
        self._seed = seed
        self.reset(seed=seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def step(self, action: int) -> Tuple[NDArray, float, bool, bool, Dict]:
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray, Dict]:
        super().reset(seed=seed)
        pass

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        else:
            raise NotImplementedError

    def _create_grid(self) -> NDArray:
        grid = np.full(self.size, ".")
        grid[self.pursuer] = "P"
        grid[self.evader] = "E"
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
