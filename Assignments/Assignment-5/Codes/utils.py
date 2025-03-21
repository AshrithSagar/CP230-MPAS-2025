"""
utils.py
Utility classes
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

Vec2 = Tuple[float, float]


class AttractiveField:
    def __init__(self, goal: Vec2, k_p: float):
        self.goal = goal
        self.k_p = k_p

    def get_potential_field(self, coord: Vec2) -> float:
        diff: Vec2 = np.array(coord) - np.array(self.goal)
        return 0.5 * self.k_p * np.dot(diff, diff)

    def get_force_field(self, coord: Vec2) -> Vec2:
        diff: Vec2 = np.array(coord) - np.array(self.goal)
        return -self.k_p * diff


class Body:
    def __init__(self, mass: float, position: Vec2, velocity: Vec2):
        self.mass = mass
        self.position: Vec2 = np.array(position, dtype=np.float32)
        self.velocity: Vec2 = np.array(velocity, dtype=np.float32)


class Obstacle(Body):
    def __init__(self, mass: float, position: Vec2, velocity: Vec2):
        super().__init__(mass, position, velocity)


class PointRobot(Body):
    def __init__(self, mass: float, position: Vec2, velocity: Vec2, vmax: float):
        super().__init__(mass, position, velocity)
        self.vmax = vmax  # Maximum horizontal velocity capability


class Scene:
    def __init__(self, time_step: float, epsilon: float = 1.0) -> None:
        self.dt = time_step
        self.epsilon = epsilon  # Coefficient of restitution
        self.bodies: List[Body] = []
        self.fig, self.ax = plt.subplots()
        self.gravity: Vec2 = np.array([0, -9.8])
        self.ground_y: float = 0

    def add_body(self, body: Body) -> None:
        self.bodies.append(body)

    def update(self) -> None:
        for body in self.bodies:
            body.velocity += self.dt * self.gravity
            body.position += self.dt * body.velocity

            # Collision with the ground
            if body.position[1] <= self.ground_y:
                body.position[1] = self.ground_y
                body.velocity[1] = -body.velocity[1]
                body.velocity[1] *= self.epsilon

    def render(self) -> None:
        def animate(_):
            self.update()
            for scatter, body in zip(self.scatters, self.bodies):
                scatter.set_offsets(body.position)

            # Dynamic axis limits
            min_x = min(body.position[0] for body in self.bodies) - 1
            max_x = max(body.position[0] for body in self.bodies) + 1
            min_y = self.ground_y - 1
            max_y = max(body.position[1] for body in self.bodies) + 1
            self.ax.set_xlim(min_x, max_x)
            self.ax.set_ylim(min_y, max_y)

        self.ax.axhline(self.ground_y, color="black", linestyle="-", linewidth=1)

        self.scatters = [self.ax.scatter(*body.position) for body in self.bodies]
        ani = FuncAnimation(self.fig, animate, frames=100, interval=50)
        plt.show()
