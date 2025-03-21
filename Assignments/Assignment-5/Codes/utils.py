"""
utils.py
Utility classes
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

Coord = Tuple[float, float]


class TwoDSpace:
    def __init__(self, dimensions: Tuple[float, float]) -> None:
        self.width, self.height = dimensions


class AttractiveField(TwoDSpace):
    def __init__(
        self, dimensions: Tuple[float, float], goal: Coord, epsilon: float
    ) -> None:
        super().__init__(dimensions)
        self.goal = goal
        self.epsilon = epsilon

    def field(self, coord: Coord) -> NDArray:
        return np.array([self.goal[0] - coord[0], self.goal[1] - coord[1]])


class PointRobot:
    def __init__(self, mass: float, position: Coord, velocity: float, vmax: float):
        self.mass = mass
        self.position: Coord = np.array(position)
        self.velocity = velocity
        self.vmax = vmax  # Maximum horizontal velocity capability
