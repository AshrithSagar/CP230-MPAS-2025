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


class Body(ABC):
    def __init__(self, mass: float, position: Vec2, velocity: Vec2):
        self.mass = mass
        self.position: Vec2 = position
        self.velocity: Vec2 = velocity
        self.pymunk_body: pymunk.Body = None

    def apply_force(self, force: Vec2) -> None:
        if self.pymunk_body:
            self.pymunk_body.apply_force_at_local_point(force)

    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        pass


class Obstacle(Body):
    def __init__(self, mass: float, position: Vec2, velocity: Vec2):
        super().__init__(mass, position, velocity)

    def draw(self, screen: pygame.Surface) -> None:
        center = (int(self.position[0]), int(self.position[1]))
        pygame.draw.rect(screen, (255, 0, 0), (*center, 20, 20))


class PointRobot(Body):
    def __init__(self, mass: float, position: Vec2, velocity: Vec2, vmax: float):
        super().__init__(mass, position, velocity)
        self.vmax = vmax  # Maximum horizontal velocity capability

    def update_velocity(self) -> None:
        speed = np.linalg.norm(self.velocity)
        if speed > self.vmax:
            self.velocity = self.velocity / speed * self.vmax

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(screen, (255, 255, 255), self.position, 10)


class Scene:
    def __init__(
        self,
        time_step: float = 0.1,
        elasticity: float = 1.0,
        display_size: Union[Tuple[int, int], str] = "full",
    ):
        self.dt = time_step
        self.elasticity = elasticity  # Coefficient of restitution
        self.size = display_size
        self.bodies: List[Body] = []

        self.space = pymunk.Space()
        self.space.gravity = (0, 9.8)
        self.ground_y = 590  # Ground level
        self._create_ground()

        self.screen = None
        self.draw_options = None

        pygame.init()
        if self.size == "full":
            display_params = {"size": (0, 0), "flags": pygame.FULLSCREEN}
        elif isinstance(self.size, tuple):
            display_params = {"size": self.size, "flags": pygame.RESIZABLE}
        self.screen = pygame.display.set_mode(**display_params)

    def _create_ground(self):
        ground = pymunk.Segment(
            self.space.static_body, (-1000, self.ground_y), (1000, self.ground_y), 1
        )
        ground.elasticity = self.elasticity
        self.space.add(ground)

    def add_body(self, body: Body) -> None:
        pymunk_body = pymunk.Body(body.mass, pymunk.moment_for_circle(body.mass, 0, 10))
        pymunk_body.position = tuple(body.position)
        pymunk_body.velocity = tuple(body.velocity)
        shape = pymunk.Circle(pymunk_body, 10)
        shape.elasticity = self.elasticity
        self.space.add(pymunk_body, shape)
        body.pymunk_body = pymunk_body
        self.bodies.append(body)

    def apply_field(self, field: AttractiveField, body: Body) -> None:
        force = field.get_force_field(body.position)
        body.apply_force(force)
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

            self.screen.fill((0, 0, 0))
            pygame.draw.line(
                self.screen, (255, 255, 255), (0, self.ground_y), (800, self.ground_y)
            )
            for body in self.bodies:
                body.position = body.pymunk_body.position
                body.draw(self.screen)
            self.space.step(self.dt)
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()
