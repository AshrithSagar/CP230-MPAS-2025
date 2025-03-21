"""
utils.py
Utility classes
"""

import os
from abc import ABC, abstractmethod
from typing import Union

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

COLORS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
}


class AttractiveField:
    def __init__(self, goal: Vec2d, k_p: float):
        self.goal = goal
        self.k_p = k_p

    def get_potential_field(self, coord: Vec2d) -> float:
        diff = coord - self.goal
        return 0.5 * self.k_p * diff.dot(diff)

    def get_force_field(self, coord: Vec2d) -> Vec2d:
        diff = coord - self.goal
        return -self.k_p * diff


class Body(pymunk.Body, ABC):
    def __init__(
        self,
        position: Vec2d,
        velocity: Vec2d = Vec2d.zero(),
        mass: float = 0,
        moment: float = pymunk.moment_for_circle(0, 0, 5),
        body_type: int = pymunk.Body.DYNAMIC,
    ):
        super().__init__(mass, moment, body_type)
        self.position = position
        self.velocity = velocity
        self.shape: pymunk.Shape = None

    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        pass


class Obstacle(Body):
    def __init__(
        self,
        position: Vec2d,
        velocity: Vec2d = Vec2d.zero(),
        mass: float = 0,
        moment: float = 0,
        body_type: int = pymunk.Body.STATIC,
        radius: float = 15,
    ):
        super().__init__(position, velocity, mass, moment, body_type)
        self.radius = radius
        self.shape = pymunk.Circle(self, self.radius)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(
            screen,
            COLORS["RED"],
            (int(self.position.x), int(self.position.y)),
            self.radius,
        )


class PointRobot(Body):
    def __init__(
        self,
        position: Vec2d,
        velocity: Vec2d = Vec2d(0, 0),
        mass: float = 1,
        vmax: float = 10,
        radius: float = 3,
    ):
        self.radius = radius
        moment = pymunk.moment_for_circle(mass, 0, self.radius)
        super().__init__(position, velocity, mass, moment)
        self.vmax = vmax  # Maximum horizontal velocity capability
        self.shape = pymunk.Circle(self, self.radius)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(
            screen,
            COLORS["GREEN"],
            (int(self.position.x), int(self.position.y)),
            self.radius,
        )

    def update_velocity(self) -> None:
        speed = self.velocity.length
        if speed > self.vmax:
            self.velocity = self.velocity.normalized() * self.vmax


class Scene:
    def __init__(
        self,
        display_size: Union[tuple[int, int], str] = "full",
        elasticity: float = 1.0,
        dt: float = 0.1,
        steps: int = 10,
    ):
        self.size = display_size
        self.elasticity = elasticity  # Coefficient of restitution
        self.dt = dt  # Time step
        self.steps = steps  # Number of steps per frame
        self.bodies: list[Body] = []

        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, 9.8)
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

    def add_bodies(self, bodies: list[Body]) -> None:
        for body in bodies:
            self.add_body(body)

    def apply_field(self, field: AttractiveField, body: Body) -> None:
        force = body.mass * field.get_force_field(body.position)
        body.apply_force_at_local_point(force, (0, 0))
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
