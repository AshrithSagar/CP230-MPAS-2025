"""
utils.py
Utility classes
"""

import math
import os
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Callable, List, Optional, Tuple, Union

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

Vec2 = Tuple[float, float]


class Colors:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    PURPLE = (255, 0, 255)


class PotentialField(ABC):
    draw_below: bool

    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        pass

    @abstractmethod
    def get_potential_field(self, coord: Vec2d) -> float:
        pass

    @abstractmethod
    def get_force_field(self, coord: Vec2d) -> Vec2d:
        pass


class Body(pymunk.Body, ABC):
    def __init__(
        self,
        position: Union[Vec2, Vec2d],
        velocity: Union[Vec2, Vec2d] = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 0,
        moment: float = pymunk.moment_for_circle(0, 0, 5),
        body_type: int = pymunk.Body.DYNAMIC,
    ):
        super().__init__(mass, moment, body_type)
        self.position = Vec2d(*position)
        self.velocity = Vec2d(*velocity)
        self.field = field
        self.shape: pymunk.Shape = None

    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        pass

    def step(self) -> None:
        pass


class Goal(Body):
    def __init__(
        self,
        position: Vec2d,
        velocity: Union[Vec2, Vec2d] = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 1,
        body_type: int = pymunk.Body.DYNAMIC,
        radius: float = 3,
    ):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        super().__init__(position, velocity, field, mass, moment, body_type)
        self.radius = radius
        self.shape = pymunk.Circle(self, radius)

    def draw(self, screen: pygame.Surface) -> None:
        if self.field is not None and self.field.draw_below:
            self.field.draw(screen)
        pygame.draw.circle(screen, Colors.PURPLE, self.position.int_tuple, self.radius)
        if self.field is not None and not self.field.draw_below:
            self.field.draw(screen)


class Obstacle(Body):
    def __init__(
        self,
        position: Union[Vec2, Vec2d],
        velocity: Union[Vec2, Vec2d] = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 0,
        moment: float = 0,
        body_type: int = pymunk.Body.STATIC,
        radius: float = 3,
    ):
        super().__init__(position, velocity, field, mass, moment, body_type)
        self.radius = radius
        self.shape = pymunk.Circle(self, radius)

    def draw(self, screen: pygame.Surface) -> None:
        if self.field is not None and self.field.draw_below:
            self.field.draw(screen)
        pygame.draw.circle(screen, Colors.RED, self.position.int_tuple, self.radius)
        if self.field is not None and not self.field.draw_below:
            self.field.draw(screen)


class StaticObstacle(Obstacle):
    def __init__(
        self,
        position: Union[Vec2, Vec2d],
        field: Optional[PotentialField] = None,
        radius: float = 3,
    ):
        super().__init__(
            position, field=field, body_type=pymunk.Body.STATIC, radius=radius
        )


class MovingObstacle(Obstacle):
    def __init__(
        self,
        position: Union[Vec2, Vec2d],
        velocity: Union[Vec2, Vec2d] = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 1,
        radius: float = 3,
    ):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        super().__init__(
            position, velocity, field, mass, moment, pymunk.Body.KINEMATIC, radius
        )


class Tunnel(Body):
    class Orient(IntEnum):
        HORIZONTAL = 0
        VERTICAL = 1

    def __init__(
        self,
        position: Union[Vec2, Vec2d],
        dimensions: Union[Vec2, Vec2d],
        orientation: Union[int, Orient] = Orient.HORIZONTAL,
        thickness: int = 3,
        field: Optional[PotentialField] = None,
    ):
        moment = pymunk.moment_for_box(0, dimensions)
        super().__init__(
            position, field=field, moment=moment, body_type=pymunk.Body.STATIC
        )
        self.dimensions = Vec2d(*dimensions)
        self.orientation = self.Orient(orientation)
        self.thickness = thickness

        hd = self.dimensions / 2
        segments = [
            [((-hd.x, -hd.y), (hd.x, -hd.y)), ((hd.x, hd.y), (-hd.x, hd.y))],
            [((-hd.x, -hd.y), (-hd.x, hd.y)), ((hd.x, hd.y), (hd.x, -hd.y))],
        ][self.orientation]
        self.segments = [pymunk.Segment(self, *seg, thickness) for seg in segments]

    def draw(self, screen: pygame.Surface) -> None:
        if self.field is not None and self.field.draw_below:
            self.field.draw(screen)
        for segment in self.segments:
            start_pos = Vec2d(*(self.position + segment.a)).int_tuple
            end_pos = Vec2d(*(self.position + segment.b)).int_tuple
            pygame.draw.line(screen, Colors.RED, start_pos, end_pos, self.thickness)
        if self.field is not None and not self.field.draw_below:
            self.field.draw(screen)

    def _is_inside(self, coord: Vec2d) -> bool:
        hd = self.dimensions / 2
        return (
            self.position.x - hd.x <= coord.x <= self.position.x + hd.x
            and self.position.y - hd.y <= coord.y <= self.position.y + hd.y
        )


class PointRobot(Body):
    def __init__(
        self,
        position: Vec2d,
        velocity: Union[Vec2, Vec2d] = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 1,
        vmax: float = 10,
        radius: float = 3,
    ):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        super().__init__(position, velocity, field, mass, moment)
        self.vmax = vmax  # Maximum horizontal velocity capability
        self.radius = radius
        self.shape = pymunk.Circle(self, radius)

    def draw(self, screen: pygame.Surface) -> None:
        if self.field is not None and self.field.draw_below:
            self.field.draw(screen)
        pygame.draw.circle(screen, Colors.BLACK, self.position.int_tuple, self.radius)
        if self.field is not None and not self.field.draw_below:
            self.field.draw(screen)

    def step(self) -> None:
        if self.velocity.x > self.vmax:
            self.velocity = self.velocity.normalized() * self.vmax


class AttractiveField(PotentialField):
    def __init__(self, k_p: float, body: Optional[Body] = None):
        self.k_p = k_p
        self.body = body
        self.draw_below: bool = True

    def draw(self, screen: pygame.Surface) -> None:
        pass

    def get_potential_field(self, coord: Vec2d) -> float:
        diff = coord - self.body.position
        return 0.5 * self.k_p * diff.dot(diff)

    def get_force_field(self, coord: Vec2d) -> Vec2d:
        diff = coord - self.body.position
        return -self.k_p * diff


class RepulsiveField(PotentialField):
    def __init__(self, k_r: float, d0: float, body: Optional[Body] = None):
        self.k_r = k_r
        self.d0 = d0  # Virtual periphery radius
        self.body = body
        self.draw_below: bool = True

    def draw(self, screen: pygame.Surface) -> None:
        d0v = self.d0 * Vec2d.ones()
        surface = pygame.Surface((2 * d0v).int_tuple, pygame.SRCALPHA)
        pygame.draw.circle(surface, (*Colors.YELLOW, 96), d0v.int_tuple, self.d0)
        screen.blit(surface, (self.body.position - d0v).int_tuple)

    def get_potential_field(self, coord: Vec2d) -> float:
        diff = coord - self.body.position
        d = diff.length
        if d >= self.d0 or d <= 1e-6:
            return 0.0
        return 0.5 * self.k_r * (1 / d - 1 / self.d0) ** 2

    def get_force_field(self, coord: Vec2d) -> Vec2d:
        diff = coord - self.body.position
        d = diff.length
        if d >= self.d0 or d <= 1e-6:
            return Vec2d.zero()
        force_mag = self.k_r * (1 / d - 1 / self.d0) / (d**2)
        if math.isnan(force_mag) or math.isnan(diff.length):
            return Vec2d.zero()
        return force_mag * diff.normalized()


class TunnelField(PotentialField):
    def __init__(self, strength: float, body: Tunnel):
        self.body = body
        self.strength = strength
        self.draw_below: bool = True

    def draw(self, screen: pygame.Surface) -> None:
        size = self.body.dimensions.int_tuple
        surface = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.rect(surface, (*Colors.YELLOW, 96), (0, 0, *size))
        dest = Vec2d(*(self.body.position - self.body.dimensions / 2)).int_tuple
        screen.blit(surface, dest)

    def get_potential_field(self, coord: Vec2d) -> float:
        if self.body._is_inside(coord):
            return self.strength
        return 0.0

    def get_force_field(self, coord: Vec2d) -> Vec2d:
        if self.body._is_inside(coord):
            forces = [Vec2d(self.strength, 0), Vec2d(0, self.strength)]
            return forces[self.body.orientation]
        return Vec2d.zero()


class Scene:
    def __init__(
        self,
        display_size: Union[Tuple[int, int], str] = "full",
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
        self.bodies: List[Body] = []
        self.fields: List[PotentialField] = []
        self.effects: dict[Body, List[PotentialField]] = {}
        self.pipeline: List[Callable] = []

        pygame.init()
        if self.size == "full":
            display_params = {"size": (0, 0), "flags": pygame.FULLSCREEN}
        elif isinstance(self.size, Tuple):
            display_params = {"size": self.size, "flags": pygame.RESIZABLE}
        self.screen: pygame.Surface = pygame.display.set_mode(**display_params)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, 9.8)
        if self.ground_y is None:
            self.ground_y = self.screen.get_height() - 50
        self.ground = pymunk.Segment(
            self.space.static_body,
            (0, self.ground_y),
            (self.screen.get_width(), self.ground_y),
            1,
        )
        self.ground.elasticity = self.elasticity
        self.space.add(self.ground)

    def add_bodies(self, bodies: List[Body]) -> None:
        for body in bodies:
            self.bodies.append(body)
            if body.shape is not None:
                body.shape.elasticity = self.elasticity
                self.space.add(body, body.shape)
            else:
                self.space.add(body)

    def add_fields(self, fields: List[PotentialField]) -> None:
        self.fields.extend(fields)

    def attach_effects(self, body: Body, fields: List[PotentialField]) -> None:
        self.effects[body] = fields

    def detach_effects(self, body: Body, fields: List[PotentialField]) -> None:
        for field in fields:
            if field in self.effects[body]:
                self.effects[body].remove(field)

    def apply_effects(self) -> None:
        for body, fields in self.effects.items():
            total_force = Vec2d.zero()
            for field in fields:
                force = field.get_force_field(body.position)
                if math.isfinite(force.length):
                    total_force += force * body.mass
            if math.isfinite(total_force.length):
                body.apply_impulse_at_local_point(total_force, (0, 0))

    def add_pipelines(self, funcs: List[Callable]) -> None:
        self.pipeline.extend(funcs)

    def render(self, stopping: Optional[Callable[[], bool]] = None) -> None:
        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            if stopping is not None and stopping():
                running = False

            for func in self.pipeline:
                func()
            self.apply_effects()
            self.screen.fill(Colors.WHITE)
            pygame.draw.line(
                self.screen,
                Colors.BLACK,
                (0, self.ground_y),
                (self.screen.get_width(), self.ground_y),
                width=1,
            )
            for field in self.fields:
                if field.draw_below:
                    field.draw(self.screen)
            for body in self.bodies:
                body.draw(self.screen)
                body.step()
            for field in self.fields:
                if not field.draw_below:
                    field.draw(self.screen)
            pygame.draw.rect(
                self.screen,
                Colors.WHITE,
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
