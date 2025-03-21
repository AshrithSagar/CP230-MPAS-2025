"""
utils.py
Utility classes
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

Vec2 = Tuple[float, float]


class TwoDSpace:
    def __init__(self, dimensions: Tuple[float, float]) -> None:
        self.width, self.height = dimensions


class AttractiveField(TwoDSpace):
    def __init__(
        self, dimensions: Tuple[float, float], goal: Vec2, epsilon: float
    ) -> None:
        super().__init__(dimensions)
        self.goal = goal
        self.epsilon = epsilon

    def field(self, coord: Vec2) -> NDArray:
        return np.array([self.goal[0] - coord[0], self.goal[1] - coord[1]])


class Body:
    def __init__(self, mass: float, position: Vec2, velocity: Vec2):
        self.mass = mass
        self.position: Vec2 = np.array(position)
        self.velocity: Vec2 = np.array(velocity)


class Obstacle(Body):
    def __init__(self, mass: float, position: Vec2, velocity: Vec2):
        super().__init__(mass, position, velocity)


class PointRobot(Body):
    def __init__(self, mass: float, position: Vec2, velocity: Vec2, vmax: float):
        super().__init__(mass, position, velocity)
        self.vmax = vmax  # Maximum horizontal velocity capability


class Scene:
    def __init__(self, time_step: float):
        self.dt = time_step
