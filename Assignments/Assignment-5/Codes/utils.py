"""
utils.py
Utility classes
"""

import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util

Vec2 = Tuple[float, float]
COLORS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
}


class AttractiveField:
    def __init__(self, goal: Vec2, k_p: float):
        self.goal = goal
        self.k_p = k_p

    def get_potential_field(self, coord: Vec2) -> float:
        diff: Vec2 = np.array(coord) - np.array(self.goal)
        return 0.5 * self.k_p * np.dot(diff, diff)

    def get_force_field(self, coord: Vec2) -> Vec2:
        diff: Vec2 = np.array(coord) - np.array(self.goal)
        return tuple(-self.k_p * np.array(diff))


class Body(pymunk.Body, ABC):
    def __init__(
        self,
        position: Vec2,
        velocity: Vec2 = (0, 0),
        mass: float = 0,
        moment: float = pymunk.moment_for_circle(0, 0, 5),
        body_type: int = pymunk.Body.DYNAMIC,
    ):
        super().__init__(mass, moment, body_type)
        self.position: Vec2 = position
        self.velocity: Vec2 = velocity
        self.shape: pymunk.Shape = None

    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        pass


class Obstacle(Body):
    def __init__(
        self,
        position: Vec2,
        velocity: Vec2 = (0, 0),
        mass: float = 0,
        moment: float = 0,
        body_type: int = pymunk.Body.STATIC,
        radius: float = 15,
    ):
        super().__init__(position, velocity, mass, moment, body_type)
        self.radius = radius
        self.shape = pymunk.Circle(self, self.radius)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(screen, COLORS["RED"], self.position, self.radius)


class PointRobot(Body):
    def __init__(
        self, position: Vec2, velocity: Vec2 = (0, 0), mass: float = 1, vmax: float = 10
    ):
        moment = pymunk.moment_for_circle(mass, 0, 5)
        super().__init__(position, velocity, mass, moment)
        self.vmax = vmax  # Maximum horizontal velocity capability
        self.shape = pymunk.Circle(self, 5)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(screen, COLORS["GREEN"], self.position, 5)

    def update_velocity(self) -> None:
        speed = np.linalg.norm(self.velocity)
        if speed > self.vmax:
            self.velocity = tuple(np.array(self.velocity) / speed * self.vmax)


class Scene:
    def __init__(
        self,
        display_size: Union[Tuple[int, int], str] = "full",
        elasticity: float = 1.0,
        dt: float = 0.1,
        steps: int = 10,
    ):
        self.size = display_size
        self.elasticity = elasticity  # Coefficient of restitution
        self.dt = dt  # Time step
        self.steps = steps  # Number of steps per frame
        self.bodies: List[Body] = []

        self.space = pymunk.Space()
        self.space.gravity = (0, 9.8)
        self.ground_y = 590  # Ground level
        self.ground = pymunk.Segment(
            self.space.static_body, (0, self.ground_y), (1000, self.ground_y), 1
        )
        self.ground.elasticity = self.elasticity
        self.space.add(self.ground)

        pygame.init()
        if self.size == "full":
            display_params = {"size": (0, 0), "flags": pygame.FULLSCREEN}
        elif isinstance(self.size, tuple):
            display_params = {"size": self.size, "flags": pygame.RESIZABLE}
        self.screen: pygame.Surface = pygame.display.set_mode(**display_params)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def add_body(self, body: Body) -> None:
        self.bodies.append(body)
        body.shape.elasticity = self.elasticity
        self.space.add(body, body.shape)

    def add_bodies(self, bodies: List[Body]) -> None:
        for body in bodies:
            self.add_body(body)

    def apply_field(self, field: AttractiveField, body: Body) -> None:
        force = field.get_force_field(body.position)
        scaled_force = pymunk.Vec2d(*tuple(f * body.mass for f in force))
        body.apply_force_at_local_point(scaled_force, (0, 0))
        if isinstance(body, PointRobot):
            body.update_velocity()

    def render(self) -> None:
        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            self.screen.fill(COLORS["BLACK"])
            pygame.draw.line(
                self.screen, COLORS["WHITE"], (0, self.ground_y), (1000, self.ground_y)
            )
            for body in self.bodies:
                body.draw(self.screen)
            for _ in range(self.steps):
                self.space.step(self.dt / self.steps)
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()
