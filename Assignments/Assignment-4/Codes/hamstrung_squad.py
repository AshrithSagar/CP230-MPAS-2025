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
        pursuer_pos: Optional[Coord] = None,
        evader_pos: Optional[Coord] = None,
        pursuer_dir: Optional[int] = None,
        store: bool = False,
    ) -> ObsType:
        _get = lambda x, default: x if x is not None else default
        pursuer_pos = _get(pursuer_pos, self.pursuer_pos)
        evader_pos = _get(evader_pos, self.evader_pos)
        pursuer_dir = _get(pursuer_dir, self.pursuer_dir)
        if store:
            self.pursuer_pos = pursuer_pos
            self.evader_pos = evader_pos
            self.pursuer_dir = pursuer_dir
        return (*pursuer_pos, *evader_pos, pursuer_dir)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict]:
        super().reset(seed=seed)
        self.pursuer_pos: Coord = (self.grid_size - 1, 0)
        self.pursuer_dir: int = 0  # 0: Up, 1: Right, 2: Down, 3: Left
        assert (
            options is not None and "evader_pos" in options
        ), "Evader position required"
        self.evader_pos: Coord = options["evader_pos"]
        info = {
            "pursuer_pos": self.pursuer_pos,
            "evader_pos": self.evader_pos,
            "pursuer_dir": self.pursuer_dir,
            "seed": self._seed,
        }
        return self._get_obs(), info

    def step(
        self, action: ActType, obs: Optional[ObsType] = None, simulate: bool = False
    ) -> Tuple[ObsType, float, bool, bool, Dict]:
        pursuer_action, evader_action = action
        if obs is not None:
            simulate = True
            pursuer_pos, evader_pos, pursuer_dir = obs[:2], obs[2:4], obs[4]
        else:
            pursuer_pos = self.pursuer_pos
            evader_pos = self.evader_pos
            pursuer_dir = self.pursuer_dir
        _safe_move = lambda pos, delta: np.clip(pos + delta, 0, self.grid_size - 1)

        # Pursuer moves
        if pursuer_action == 1:  # Turn right
            pursuer_dir = (pursuer_dir + 1) % 4
        pursuer_delta = np.array([(-2, 0), (0, 2), (2, 0), (0, -2)][pursuer_dir])
        pursuer_pos = _safe_move(pursuer_pos, pursuer_delta)

        # Evader moves
        evader_delta = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)][evader_action])
        evader_pos = _safe_move(evader_pos, evader_delta)

        # Check termination
        terminated: bool = np.linalg.norm(pursuer_pos - evader_pos) <= 1.5
        obs = self._get_obs(pursuer_pos, evader_pos, pursuer_dir, store=not simulate)
        return obs, 0.0, terminated, terminated, {}

    def render(self) -> None:
        if self.render_mode == "ansi":
            return self._render_ansi()
        else:
            raise NotImplementedError

    def _create_grid(self) -> NDArray:
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        if hasattr(self, "evader_pos"):
            grid[tuple(self.evader_pos)] = "E"
        if hasattr(self, "pursuer_pos"):
            grid[tuple(self.pursuer_pos)] = "P"
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
