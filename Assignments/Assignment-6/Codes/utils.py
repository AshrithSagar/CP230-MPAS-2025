"""
utils.py \n
Utility classes for velocity obstacle calculations.
"""

import os
import random
from typing import List, Optional

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import numpy as np
import pygame
import pymunk
from pygame.color import Color
from pymunk import Vec2d

random.seed(42)
np.random.seed(42)


class Scheme:
    """Color scheme for different elements in the simulation."""

    BACKGROUND = Color("white")
    ROBOTS = [Color("red"), Color("orange"), Color("blue")]
    CONE_ALPHA = 96


class Robot(pymunk.Body):
    """Circular robot with a position, velocity, and radius."""

    def __init__(
        self,
        position: Optional[Vec2d] = None,
        velocity: Optional[Vec2d] = None,
        radius: Optional[float] = None,
        mass: float = 1,
    ):
        self.radius = radius or np.random.uniform(1, 5)
        moment = pymunk.moment_for_circle(mass, 0, self.radius)
        super().__init__(mass, moment)
        if position is not None:
            self.position = Vec2d(*position)
        else:
            self.position = Vec2d(*np.random.uniform(0, 200, size=2))
        if velocity is not None:
            self.velocity = Vec2d(*velocity)
        else:
            speed, angle = np.random.uniform(10, 50), np.random.uniform(0, 2 * np.pi)
            self.velocity = Vec2d(speed, 0).rotated(angle)
        self.color = Scheme.ROBOTS.pop()
        self.shape = pymunk.Circle(self, self.radius)

    def __repr__(self):
        return (
            f"Robot(position={round(self.position.x, 2), round(self.position.y, 2)}, "
            f"velocity={round(self.velocity.x, 2), round(self.velocity.y, 2)}, "
            f"radius={round(self.radius, 2)}"
        )

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the robot on the screen."""
        pygame.draw.circle(
            screen,
            self.color,
            self.position.int_tuple,
            self.radius,
        )


class Scene:
    """Scene class to render the simulation environment."""

    def __init__(
        self, time_step: float = 0.1, sub_steps: int = 10, scale: bool = False
    ):
        self.dt = time_step
        self.sub_steps = sub_steps  # Number of sub-steps per time step
        self.bodies: List[Robot] = []

        pygame.init()
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE
        flags |= pygame.SCALED if scale else 0
        self.screen = pygame.display.set_mode((200, 200), flags)
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        borders = [
            ((0, 0), (200, 0)),
            ((200, 0), (200, 200)),
            ((200, 200), (0, 200)),
            ((0, 200), (0, 0)),
        ]
        for border in borders:
            wall = pymunk.Segment(self.space.static_body, *border, 0)
            wall.elasticity = 1
            self.space.add(wall)

    def add_bodies(self, bodies: List[Robot]) -> None:
        """Add bodies to the simulation environment."""
        for body in bodies:
            body.shape.elasticity = 1
            self.bodies.append(body)
            self.space.add(body, body.shape)

    def render(self, framerate: int = 60) -> None:
        """
        Start the simulation and render the environment.
        Press `Esc` or close the window to stop the simulation.
        - `framerate`: Frames per second for rendering the simulation.
        """

        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            self.screen.fill(Scheme.BACKGROUND)
            for body in self.bodies:
                body.draw(self.screen)

            # Step the simulation
            for _ in range(self.sub_steps):
                self.space.step(self.dt / self.sub_steps)
            pygame.display.flip()  # Update the display
            clock.tick(framerate)
        pygame.quit()
