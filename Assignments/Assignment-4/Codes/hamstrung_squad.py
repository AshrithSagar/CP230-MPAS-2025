"""
hamstrung_squad.py
Hamstrung squad game environment
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

Coord = Tuple[int, int]
ActType = Tuple[int, int]
ObsType = Tuple[int, int, int, int, int]


class HamstrungSquadEnv(gym.Env):
    """Hamstrung squad game environment"""

    metadata = {"render.modes": ["ansi"], "render.fps": 4}

    def __init__(
        self,
        grid_size: int = 10,
        render_mode: str = "ansi",
        seed: int = 42,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(2), gym.spaces.Discrete(4))
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=grid_size - 1, shape=(5,), dtype=np.int32
        )
        self.seed(seed=seed)

    def seed(self, seed: int = None) -> None:
        self._seed = seed
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def _get_obs(
        self,
        pursuer: Optional[Coord] = None,
        evader: Optional[Coord] = None,
        pursuer_direction: Optional[int] = None,
        store: bool = False,
    ) -> ObsType:
        _get = lambda x, default: x if x is not None else default
        pursuer = _get(pursuer, self.pursuer)
        evader = _get(evader, self.evader)
        pursuer_direction = _get(pursuer_direction, self.pursuer_direction)
        if store:
            self.pursuer = pursuer
            self.evader = evader
            self.pursuer_direction = pursuer_direction
        return (*pursuer, *evader, pursuer_direction)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict]:
        super().reset(seed=seed)
        self.pursuer: Coord = (self.grid_size - 1, 0)
        self.pursuer_direction: int = 0  # 0: Up, 1: Right, 2: Down, 3: Left
        assert options is not None and "evader" in options, "Evader position required"
        self.evader: Coord = options["evader"]
        info = {
            "pursuer_start": self.pursuer,
            "evader_start": self.evader,
            "pursuer_direction": self.pursuer_direction,
            "seed": self._seed,
        }
        return self._get_obs(), info

    def step(
        self, action: ActType, simulate: bool = False
    ) -> Tuple[ObsType, float, bool, bool, Dict]:
        pursuer_action, evader_action = action
        _safe_move = lambda pos, delta: np.clip(pos + delta, 0, self.grid_size - 1)

        # Pursuer moves
        if pursuer_action == 0:  # Forward
            pursuer_direction = self.pursuer_direction
        elif pursuer_action == 1:  # Turn right
            pursuer_direction = (self.pursuer_direction + 1) % 4
        pursuer_delta = np.array([(-2, 0), (0, 2), (2, 0), (0, -2)][pursuer_direction])
        pursuer = _safe_move(self.pursuer, pursuer_delta)

        # Evader moves
        evader_delta = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)][evader_action])
        evader = _safe_move(self.evader, evader_delta)

        # Check termination
        terminated: bool = np.linalg.norm(pursuer - evader) <= 1.5
        obs = self._get_obs(pursuer, evader, pursuer_direction, store=not simulate)
        return obs, 0.0, terminated, terminated, {}

    def render(self) -> None:
        if self.render_mode == "ansi":
            return self._render_ansi()
        else:
            raise NotImplementedError

    def _create_grid(self) -> NDArray:
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        if hasattr(self, "evader"):
            grid[tuple(self.evader)] = "E"
        if hasattr(self, "pursuer"):
            grid[tuple(self.pursuer)] = "P"
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
