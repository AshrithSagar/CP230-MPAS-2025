"""
hamstrung_squad.py
Hamstrung sqaud game environment
"""

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from numpy.typing import NDArray

Coord = Tuple[int, int]
Block = List[Coord]
Path = List[Coord]


class HamstrungSquadEnv(gym.Env):
    """Hamstrung squad game environment"""

    metadata = {"render.modes": ["ansi"]}

    def __init__(
        self,
        size: Tuple[int, int],
        start: Coord,
        goal: Block,
        max_steps: int = 10,
        render_mode: str = "ansi",
        seed: int = 42,
    ):
        super().__init__()
        self.size = size
        self.start = start
        self.goal = goal
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
        pass
