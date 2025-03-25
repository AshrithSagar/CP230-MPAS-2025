"""
utils.py \n
Utility classes for velocity obstacle calculations.
"""

import math
import os
import random
from typing import List, Optional

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import pymunk
from pygame.color import Color
from pymunk import Vec2d

random.seed(24233)


class Scheme:
    """Color scheme for different elements in the simulation."""

    BACKGROUND = Color("white")
    ROBOTS = [Color("red"), Color("orange"), Color("blue")]
    CONE_ALPHA = 96


class Robot:
    """Circular robot with a position, velocity, and radius."""

    def __init__(
        self,
        position: Optional[Vec2d] = None,
        velocity: Optional[Vec2d] = None,
        radius: Optional[float] = None,
        mass: float = 1,
    ):
        self.radius = radius or random.uniform(1, 5)
        moment = pymunk.moment_for_circle(mass, 0, self.radius)
        self.body = pymunk.Body(mass, moment)
        if position is not None:
            self.body.position = Vec2d(*position)
        else:
            self.body.position = Vec2d(random.uniform(0, 200), random.uniform(0, 200))
        if velocity is not None:
            self.body.velocity = Vec2d(*velocity)
        else:
            speed, angle = random.uniform(10, 50), random.uniform(0, 2 * math.pi)
            self.body.velocity = Vec2d(speed, 0).rotated(angle)
        self.color = Scheme.ROBOTS.pop()
        self.shape = pymunk.Circle(self.body, self.radius)

    def __repr__(self):
        return (
            f"Robot(position={round(self.body.position.x, 2), round(self.body.position.y, 2)}, "
            f"velocity={round(self.body.velocity.x, 2), round(self.body.velocity.y, 2)}, "
            f"radius={round(self.radius, 2)}"
        )

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the robot on the screen."""
        pygame.draw.circle(
            screen, self.color, Vec2d(*self.body.position).int_tuple, self.radius
        )


class VelocityObstacle:
    """
    Velocity obstacle for two robots to avoid collision. \n
    The velocity obstacle is a cone that represents the possible velocities for robotA to avoid colliding with robotB.
    """

    def __init__(self, robotA: Robot, robotB: Robot):
        self.robotA = robotA
        self.robotB = robotB
        self.radius_sum = robotA.radius + robotB.radius

    @property
    def rel_pos(self) -> Vec2d:
        """Relative position of robotB with respect to robotA."""
        return Vec2d(*(self.robotB.body.position - self.robotA.body.position))

    @property
    def rel_vel(self) -> Vec2d:
        """Relative velocity of robotB with respect to robotA."""
        return Vec2d(*(self.robotB.body.velocity - self.robotA.body.velocity))

    def compute(self) -> List[Vec2d]:
        """Compute the velocity obstacle for two robots."""
        vo = []
        d = self.rel_pos.length
        if self.rel_vel.get_length_sqrd() < 1e-6:
            return vo

        # If already colliding, use a simple avoidance direction
        if d < self.radius_sum:
            vo.append(self.rel_vel.normalized())
        else:
            half_angle = math.asin(self.radius_sum / d)
            base_direction = self.rel_pos.normalized()
            tangent1 = base_direction.rotated(half_angle)
            tangent2 = base_direction.rotated(-half_angle)
            vo.extend([tangent1, tangent2])
        return vo

    def draw(self, screen: pygame.Surface):
        """
        Draw a filled translucent cone representing the velocity obstacle.
        The cone is drawn with vertex at robotA's position and spanning the two tangent directions.
        """
        directions = self.compute()
        if len(directions) < 2:
            return

        start = Vec2d(*self.robotA.body.position)
        angles = [direction.angle % (2 * math.pi) for direction in directions]
        angle1, angle2 = min(angles), max(angles)

        # Sample intermediate points along the arc from angle1 to angle2.
        arc_points, num_points = [], 30
        length = self.rel_pos.length - self.robotB.radius
        for i in range(num_points + 1):
            theta = angle1 + (angle2 - angle1) * i / num_points
            point = start + Vec2d(math.cos(theta), math.sin(theta)) * length
            arc_points.append(point.int_tuple)

        polygon_points = [start.int_tuple] + arc_points
        color = self.robotB.color
        color.a = Scheme.CONE_ALPHA
        surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(surface, color, polygon_points)
        screen.blit(surface, (0, 0))


class Scene:
    """Scene class to render the simulation environment."""

    def __init__(
        self, time_step: float = 0.1, sub_steps: int = 10, scale: bool = False
    ):
        self.dt = time_step
        self.sub_steps = int(sub_steps)  # Number of sub-steps per time step

        self.bodies: List[Robot] = []
        self.vos: List[VelocityObstacle] = []

        pygame.init()
        pygame.display.set_caption("Velocity obstacles")
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
            wall.elasticity = 1.0
            self.space.add(wall)

    def add_bodies(self, robots: List[Robot]) -> None:
        """Add bodies to the simulation environment."""
        for robot in robots:
            robot.shape.elasticity = 1.0
            self.bodies.append(robot)
            self.space.add(robot.body, robot.shape)

    def add_vos(self, vos: List[VelocityObstacle]) -> None:
        """Add velocity obstacles to the simulation environment."""
        self.vos.extend(vos)

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
            for vo in self.vos:
                vo.draw(self.screen)
            for body in self.bodies:
                body.draw(self.screen)

            # Step the simulation
            for _ in range(self.sub_steps):
                self.space.step(self.dt / self.sub_steps)
            pygame.display.flip()  # Update the display
            clock.tick(framerate)
        pygame.quit()
