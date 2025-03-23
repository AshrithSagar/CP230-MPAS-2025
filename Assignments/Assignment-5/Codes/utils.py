"""
utils.py
Utility classes for simulation of potential fields and bodies in a 2D environment.
"""

import math
import os
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple, Union

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

# Type aliases
Vec2 = Tuple[float, float]
Color = Tuple[int, int, int]


class Colors:
    """Predefined color constants."""

    WHITE: Color = (255, 255, 255)
    BLACK: Color = (0, 0, 0)
    RED: Color = (255, 0, 0)
    GREEN: Color = (0, 255, 0)
    BLUE: Color = (0, 0, 255)
    YELLOW: Color = (255, 255, 0)
    PURPLE: Color = (255, 0, 255)
    ORANGE: Color = (255, 165, 0)


class Scheme:
    """Color scheme for different elements in the simulation."""

    BACKGROUND = Colors.WHITE
    GROUND = Colors.BLACK
    GOAL = Colors.PURPLE
    OBSTACLE = Colors.RED
    ROBOT = Colors.BLUE
    FIELD = (*Colors.YELLOW, 96)  # Semi-transparent yellow


class PotentialField(ABC):
    """Abstract base class for potential fields."""

    def __init__(self, draw_below: bool = True):
        """
        Initialize the potential field with the given parameters.
        - `draw_below`: If True, the field is drawn below it's associated body, else above it.
        """
        self.draw_below = draw_below

    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the potential field on the screen. \n
        This method should be implemented in the derived classes to visualize the field in the simulation.
        """
        pass

    @abstractmethod
    def get_potential_field(self, coord: Vec2d) -> float:
        """Calculate the potential field value at a given coordinate."""
        pass

    @abstractmethod
    def get_force_field(self, coord: Vec2d) -> Vec2d:
        """Calculate the force vector at a given coordinate."""
        pass


class Body(pymunk.Body, ABC):
    """Abstract base class for rigid bodies in the simulation."""

    def __init__(
        self,
        position: Union[Vec2, Vec2d],
        velocity: Union[Vec2, Vec2d] = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 0,
        moment: float = pymunk.moment_for_circle(0, 0, 5),
        body_type: int = pymunk.Body.DYNAMIC,
    ):
        """
        Initialize the rigid body with the given parameters.
        - `position`: Initial position of the body.
        - `velocity`: Initial velocity of the body.
        - `field`: Associated potential field for the body.
        - `mass`: Mass of the body.
        - `moment`: Moment of inertia of the body.
        - `body_type`: Type of the body (static, dynamic or kinematic).
        """
        super().__init__(mass, moment, body_type)
        self.position = Vec2d(*position)  # Initial position
        self.velocity = Vec2d(*velocity)  # Initial velocity
        self.field = field  # Associated potential field
        self.shape: pymunk.Shape = None  # Body shape for collision detection

    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the body on the screen. \n
        This method should be implemented in the derived classes.
        """
        pass

    def step(self) -> None:
        """Function to update any state variables of the body at each time step."""
        pass


class Goal(Body):
    """
    Represents a goal in the simulation. \n
    Attracts other bodies using an attractive potential field.
    """

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
        pygame.draw.circle(screen, Scheme.GOAL, self.position.int_tuple, self.radius)
        if self.field is not None and not self.field.draw_below:
            self.field.draw(screen)


class Obstacle(Body):
    """
    Represents an obstacle in the simulation. \n
    Can be static or moving (dynamic or kinematic), and has a repulsive potential field.
    """

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
        pygame.draw.circle(
            screen, Scheme.OBSTACLE, self.position.int_tuple, self.radius
        )
        if self.field is not None and not self.field.draw_below:
            self.field.draw(screen)


class StaticObstacle(Obstacle):
    """
    Represents a static obstacle in the simulation. \n
    A static obstacle does not move and has a repulsive potential field.
    """

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
    """
    Represents a moving obstacle in the simulation. \n
    A moving obstacle has a velocity and can move around and interact with other bodies, fields and the environment.
    """

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
            position,
            velocity,
            field,
            mass,
            moment,
            body_type=pymunk.Body.DYNAMIC,
            radius=radius,
        )


class Tunnel(Body):
    """
    Represents a tunnel in the simulation. \n
    A tunnel is a rectangular region that bodies can pass through.
    """

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

        # Segments (walls) of the tunnel
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
            pygame.draw.line(
                screen, Scheme.OBSTACLE, start_pos, end_pos, self.thickness
            )
        if self.field is not None and not self.field.draw_below:
            self.field.draw(screen)

    def _is_inside(self, coord: Vec2d) -> bool:
        """Check if a coordinate is inside the tunnel."""
        hd = self.dimensions / 2
        return (
            self.position.x - hd.x <= coord.x <= self.position.x + hd.x
            and self.position.y - hd.y <= coord.y <= self.position.y + hd.y
        )


class PointRobot(Body):
    """
    Represents a point robot in the simulation. \n
    The robot can move and interact with potential fields and other bodies.
    """

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
        pygame.draw.circle(screen, Scheme.ROBOT, self.position.int_tuple, self.radius)
        if self.field is not None and not self.field.draw_below:
            self.field.draw(screen)

    def step(self) -> None:
        """Ensure robot's horizontal velocity does not exceed the maximum capable limit."""
        if self.velocity.x > self.vmax:
            self.velocity = self.velocity.normalized() * self.vmax


class AttractiveField(PotentialField):
    """
    Represents an radial attractive field in the simulation. \n
    Attracts bodies towards a source with a force proportional to the distance.
    """

    def __init__(self, k_p: float, body: Optional[Body] = None):
        """
        Initialize the attractive field with the given parameters. \n
        The force field at a point `x` is given by `-k_p * (x - x0)`, where `x0` is the source position.
        - `k_p`: Proportional constant for the attractive force.
        - `body`: The source body that exerts the attractive force.
        """
        super().__init__()
        self.k_p = k_p
        self.body = body

    def draw(self, screen: pygame.Surface) -> None:
        pass

    def get_potential_field(self, coord: Vec2d) -> float:
        diff = coord - self.body.position
        return 0.5 * self.k_p * diff.dot(diff)

    def get_force_field(self, coord: Vec2d) -> Vec2d:
        diff = coord - self.body.position
        return -self.k_p * diff


class RepulsiveField(PotentialField):
    """
    Represents a radial repulsive field in the simulation. \n
    Exerts a force on bodies within a certain radius (virtual periphery) around the source.
    """

    def __init__(
        self,
        k_r: float,
        d0: float,
        body: Optional[Body] = None,
    ):
        """
        Initialize the repulsive field with the given parameters. \n
        The force field at a point `x` is given by `k_r * (1/d - 1/d0) * 1 / d^2`,
        where `d` is the distance between `x` and the source position.
        - `k_r`: Repulsion constant for the repulsive force.
        - `d0`: Virtual periphery radius around the source.
        - `body`: The source body that exerts the repulsive force.
        """
        super().__init__()
        self.k_r = k_r
        self.d0 = d0  # Virtual periphery radius
        self.body = body

    def draw(self, screen: pygame.Surface) -> None:
        d0v = self.d0 * Vec2d.ones()
        surface = pygame.Surface((2 * d0v).int_tuple, pygame.SRCALPHA)
        pygame.draw.circle(surface, Scheme.FIELD, d0v.int_tuple, self.d0)
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
    """
    Represents a tunnel field in the simulation. \n
    Exerts a force on bodies passing through it.
    """

    def __init__(self, strength: float, body: Tunnel):
        super().__init__()
        self.strength = strength
        self.body = body

    def draw(self, screen: pygame.Surface) -> None:
        size = self.body.dimensions.int_tuple
        surface = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.rect(surface, Scheme.FIELD, (0, 0, *size))
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
    """
    Represents a 2D simulation environment with bodies, potential fields and effects.
    Call `render()` method to start the simulation, after adding bodies, fields and effects as desired.
    The interaction between bodies and fields is handled through effects, which can be attached (or detached) to the bodies using the `attach_effects()` (or `detach_effects()`) method. \n
    Add the bodies in the scene using the `add_bodies()` method.
    Potential fields are automatically handled by the bodies if they are associated with them, if not they can be added using the `add_fields())` method.
    Pipeline functions can be added using the `add_pipelines()` method to perform additional operations at each time step, such as for toggling effects or for updating the states of the bodies.
    """

    def __init__(
        self,
        display_size: Union[Tuple[int, int], str] = "full",
        elasticity: float = 1.0,
        ground_y: Optional[int] = None,
        time_step: float = 0.1,
        sub_steps: int = 10,
    ):
        """
        Initialize the simulation environment with the given parameters.
        - `display_size`: Size of the display window in pixels, or `"full"` for fullscreen.
        - `elasticity`: Coefficient of restitution for collisions.
        - `ground_y`: Y-coordinate of the ground level, or `None` to set it near the bottom of the display window.
            The top-left corner is the origin `(0, 0)`, with positive y-axis pointing downwards.
        - `time_step`: Time step for the simulation.
        - `sub_steps`: Number of sub-steps per time step.
            Increase this value for smoother simulations.
        """

        self.size = display_size
        self.elasticity = elasticity  # Coefficient of restitution
        self.ground_y = ground_y  # Ground level
        self.dt = time_step
        self.sub_steps = sub_steps  # Number of sub-steps per time step

        self.bodies: List[Body] = []
        self.fields: List[PotentialField] = []
        self.effects: Dict[Body, List[PotentialField]] = {}
        self.pipeline: List[Callable] = []

        # Initialize the Pygame window
        pygame.init()
        if self.size == "full":
            display_params = {"size": (0, 0), "flags": pygame.FULLSCREEN}
        elif isinstance(self.size, Tuple):
            display_params = {"size": self.size, "flags": pygame.RESIZABLE}
        self.screen: pygame.Surface = pygame.display.set_mode(**display_params)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        # Initialize the PyMunk space
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
        """Add bodies to the simulation environment."""
        for body in bodies:
            self.bodies.append(body)
            if body.shape is not None:
                body.shape.elasticity = self.elasticity
                self.space.add(body, body.shape)
            else:
                self.space.add(body)

    def add_fields(self, fields: List[PotentialField]) -> None:
        """
        Add potential fields to the simulation environment. \n
        This method should be used to add fields that are not associated with any body.
        """
        self.fields.extend(fields)

    def attach_effects(self, body: Body, fields: List[PotentialField]) -> None:
        """
        Enable the body to be affected by the given potential fields. \n
        The fields will be calculated at each time step and the total force will be applied to the body.
        """
        self.effects[body] = fields

    def detach_effects(self, body: Body, fields: List[PotentialField]) -> None:
        """Undo attach effects for the body."""
        for field in fields:
            if field in self.effects[body]:
                self.effects[body].remove(field)

    def apply_effects(self) -> None:
        """Calculate and apply the total force on each body due to the attached potential fields."""
        for body, fields in self.effects.items():
            total_force = Vec2d.zero()
            for field in fields:
                force = field.get_force_field(body.position)
                if math.isfinite(force.length):
                    total_force += force * body.mass
            if math.isfinite(total_force.length):
                body.apply_impulse_at_local_point(total_force, (0, 0))

    def add_pipelines(self, funcs: List[Callable]) -> None:
        """Add functions to the pipeline to be executed at each time step before rendering the frame."""
        self.pipeline.extend(funcs)

    def render(
        self, stopping: Optional[Callable[[], bool]] = None, framerate: int = 60
    ) -> None:
        """
        Start the simulation and render the environment.
        Press `Esc` or close the window to stop the simulation.
        - `stopping`: Optional function that returns a bool. If True, the simulation will stop.
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
            if stopping is not None and stopping():
                running = False

            for func in self.pipeline:
                func()
            self.apply_effects()

            # Background and ground
            self.screen.fill(Scheme.BACKGROUND)
            pygame.draw.line(
                self.screen,
                Scheme.GROUND,
                (0, self.ground_y),
                (self.screen.get_width(), self.ground_y),
                width=1,
            )

            # Draw bodies and fields
            for field in self.fields:
                if field.draw_below:
                    field.draw(self.screen)
            for body in self.bodies:
                body.draw(self.screen)
                body.step()
            for field in self.fields:
                if not field.draw_below:
                    field.draw(self.screen)

            # Render background again below the ground
            pygame.draw.rect(
                self.screen,
                Scheme.BACKGROUND,
                (
                    0,
                    self.ground_y + 1,
                    self.screen.get_width(),
                    self.screen.get_height() - self.ground_y,
                ),
            )

            # Step the simulation
            for _ in range(self.sub_steps):
                self.space.step(self.dt / self.sub_steps)
            pygame.display.flip()  # Update the display
            clock.tick(framerate)
        pygame.quit()
