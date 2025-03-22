"""
utils.py
Utility classes
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Union

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
    "BLUE": (0, 0, 255),
    "YELLOW": (255, 255, 0),
}


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
        d0: float = 0,
    ):
        super().__init__(position, velocity, mass, moment, body_type)
        self.radius = radius
        self.d0 = d0
        self.shape = pymunk.Circle(self, self.radius)

    def draw(self, screen: pygame.Surface) -> None:
        if self.d0 > 0:
            surface = pygame.Surface(
                (2 * int(self.d0), 2 * int(self.d0)), pygame.SRCALPHA
            )
            pygame.draw.circle(
                surface,
                (*COLORS["YELLOW"], 96),
                (int(self.d0), int(self.d0)),
                int(self.d0),
            )
            screen.blit(
                surface,
                (int(self.position.x - self.d0), int(self.position.y - self.d0)),
            )
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
        if self.velocity.x > self.vmax:
            self.velocity = self.velocity.normalized() * self.vmax


class PotentialField(ABC):
    @abstractmethod
    def get_potential_field(self, coord: Vec2d) -> float:
        pass

    @abstractmethod
    def get_force_field(self, coord: Vec2d) -> Vec2d:
        pass


class AttractiveField(PotentialField):
    def __init__(self, goal: Vec2d, k_p: float):
        self.goal = goal
        self.k_p = k_p

    def get_potential_field(self, coord: Vec2d) -> float:
        diff = coord - self.goal
        return 0.5 * self.k_p * diff.dot(diff)

    def get_force_field(self, coord: Vec2d) -> Vec2d:
        diff = coord - self.goal
        return -self.k_p * diff


class RepulsiveField(PotentialField):
    def __init__(self, obstacle: Obstacle, k_r: float, d0: float = None):
        self.obstacle = obstacle
        self.k_r = k_r
        self.d0 = d0 or obstacle.d0  # Virtual periphery radius

    def get_potential_field(self, coord: Vec2d) -> float:
        diff = coord - self.obstacle.position
        d = diff.length
        if d >= self.d0 or d == 0:
            return 0.0
        return 0.5 * self.k_r * (1 / d - 1 / self.d0) ** 2

    def get_force_field(self, coord: Vec2d) -> Vec2d:
        diff = coord - self.obstacle.position
        d = diff.length
        if d >= self.d0 or d == 0:
            return Vec2d.zero()
        force_mag = self.k_r * (1 / d - 1 / self.d0) / (d**2)
        return force_mag * diff.normalized()


class Scene:
    def __init__(
        self,
        display_size: Union[tuple[int, int], str] = "full",
        elasticity: float = 1.0,
        ground_y: Optional[int] = None,
        dt: float = 0.1,
        steps: int = 10,
    ):
        self.size = display_size
        self.elasticity = elasticity  # Coefficient of restitution
        self.ground_y = ground_y  # Ground level
        self.dt = dt  # Time step
        self.steps = steps  # Number of steps per frame
        self.bodies: list[Body] = []
        self.fields: dict[Body, list[PotentialField]] = {}

        pygame.init()
        if self.size == "full":
            display_params = {"size": (0, 0), "flags": pygame.FULLSCREEN}
        elif isinstance(self.size, tuple):
            display_params = {"size": self.size, "flags": pygame.RESIZABLE}
        self.screen: pygame.Surface = pygame.display.set_mode(**display_params)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, 9.8)
        if self.ground_y is None:
            self.ground_y = self.screen.get_height() - 10
        self.ground = pymunk.Segment(
            self.space.static_body,
            (0, self.ground_y),
            (self.screen.get_width(), self.ground_y),
            1,
        )
        self.ground.elasticity = self.elasticity
        self.space.add(self.ground)

    def add_bodies(self, bodies: list[Body]) -> None:
        for body in bodies:
            self.bodies.append(body)
            body.shape.elasticity = self.elasticity
            self.space.add(body, body.shape)

    def attach_fields(self, body: Body, fields: list[PotentialField]) -> None:
        self.fields[body] = fields

    def apply_fields(self) -> None:
        for body, fields in self.fields.items():
            total_force = Vec2d(0, 0)
            for field in fields:
                total_force += field.get_force_field(body.position) * body.mass
            body.apply_impulse_at_local_point(total_force, (0, 0))

    def render(self) -> None:
        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            self.apply_fields()
            self.screen.fill(COLORS["BLACK"])
            pygame.draw.line(
                self.screen,
                COLORS["WHITE"],
                (0, self.ground_y),
                (self.screen.get_width(), self.ground_y),
                width=1,
            )
            for body in self.bodies:
                body.draw(self.screen)
            pygame.draw.rect(
                self.screen,
                COLORS["BLACK"],
                (
                    0,
                    self.ground_y + 1,
                    self.screen.get_width(),
                    self.screen.get_height() - self.ground_y,
                ),
            )
            for _ in range(self.steps):
                self.space.step(self.dt / self.steps)
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()
