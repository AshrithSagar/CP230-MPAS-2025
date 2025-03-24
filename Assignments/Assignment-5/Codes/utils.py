"""
utils.py \n
Utility classes for simulation of potential fields and bodies in a 2D environment.
"""

import math
import os
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple, Union

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import moviepy
import pygame
import pymunk
import pymunk.pygame_util
from pygame.color import Color
from pymunk import Vec2d

# Type aliases
Vec2 = Tuple[float, float]


class Scheme:
    """Color scheme for different elements in the simulation."""

    BACKGROUND = Color("white")
    GROUND = Color("black")
    GOAL = Color("purple")
    OBSTACLE = Color("red")
    ROBOT = Color("blue")
    FIELD = Color("yellow")
    FIELD.a = 96  # Alpha


class PotentialField(ABC):
    """
    Abstract base class for potential fields. \n
    - `draw_below`: If True, the field is drawn below it's associated body, else above it.
    """

    draw_below: bool = True

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
    """
    Abstract base class for rigid bodies in the simulation. \n
    Extends the PyMunk `Body` class with additional attributes and methods.
    Not intended to be instantiated directly, use the derived classes preferably.
    - `field`: Associated potential field for the body.
    - `shape`: Body shape for collision detection.
    - `body_type`: Type of the body (static, dynamic or kinematic).
    """

    def __init__(
        self,
        position: Vec2,
        velocity: Vec2 = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 0,
        moment: float = 0,
        body_type: int = pymunk.Body.DYNAMIC,
    ):
        super().__init__(mass, moment, body_type)
        self.position = Vec2d(*position)  # Initial position
        self.velocity = Vec2d(*velocity)  # Initial velocity
        self.field = field
        self.shape: pymunk.Shape = None

    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the body on the screen. \n
        This method should be implemented in the derived classes, by using the Pygame drawing functions.
        """
        pass

    def step(self) -> None:
        """Function to update any state variables of the body at each time step."""
        pass

    def stop(self) -> None:
        """Stop the body by setting the velocity to zero."""
        self._set_velocity((0, 0))


class PointBody(Body):
    """
    Represents a point body in the simulation. \n
    A point body here is approximated as a circle with a small radius.
    """

    def __init__(
        self,
        scheme: Color,
        position: Vec2d,
        velocity: Vec2 = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 1,
        body_type: int = Body.DYNAMIC,
        radius: float = 3,
    ):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        super().__init__(position, velocity, field, mass, moment, body_type)
        self.scheme = scheme  # Body fill color
        self.radius = radius
        self.shape = pymunk.Circle(self, radius)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(screen, self.scheme, self.position.int_tuple, self.radius)


class Goal(PointBody):
    """
    Represents a goal in the simulation. \n
    Attracts other bodies using an attractive potential field.
    """

    def __init__(
        self,
        position: Vec2d,
        velocity: Vec2 = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 1,
        body_type: int = Body.STATIC,
        vd: float = 10,
        radius: float = 3,
    ):
        super().__init__(
            Scheme.GOAL, position, velocity, field, mass, body_type, radius
        )
        self.vd = vd  # Horizontal velocity when motion is enabled

    def enable_motion(self) -> None:
        """Enable motion for the goal."""
        self.body_type = Body.KINEMATIC
        self._set_velocity((self.vd, 0))  # Simplified case


class PointObstacle(Body):
    """
    Represents an point obstacle in the simulation. \n
    Can be static or moving (dynamic or kinematic), and has a repulsive potential field.
    """

    def __init__(
        self,
        position: Vec2,
        velocity: Vec2 = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 0,
        moment: float = 0,
        body_type: int = Body.STATIC,
        radius: float = 3,
    ):
        super().__init__(position, velocity, field, mass, moment, body_type)
        self.radius = radius
        self.shape = pymunk.Circle(self, radius)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(
            screen, Scheme.OBSTACLE, self.position.int_tuple, self.radius
        )


class StaticPointObstacle(PointObstacle):
    """
    Represents a static point obstacle in the simulation. \n
    A static obstacle does not move and has a repulsive potential field.
    """

    def __init__(
        self,
        position: Vec2,
        field: Optional[PotentialField] = None,
        radius: float = 3,
    ):
        super().__init__(position, field=field, body_type=Body.STATIC, radius=radius)


class MovingPointObstacle(PointObstacle):
    """
    Represents a moving point obstacle in the simulation. \n
    A moving obstacle has a velocity and can move around and interact with other bodies, fields and the environment.
    """

    def __init__(
        self,
        position: Vec2,
        velocity: Vec2 = Vec2d.zero(),
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
            body_type=Body.DYNAMIC,
            radius=radius,
        )


class PolyObstacle(Body):
    """Represents a polygonal obstacle, defined by a list of vertices."""

    def __init__(
        self,
        vertices: List[Vec2],
        position: Vec2,
        velocity: Vec2 = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 0,
        body_type: int = Body.STATIC,
        radius: float = 0,
    ):
        self.vertices = [Vec2d(*v) for v in vertices]
        moment = pymunk.moment_for_poly(mass, self.vertices, radius=radius)
        super().__init__(position, velocity, field, mass, moment, body_type=body_type)
        self.radius = radius
        self.shape: pymunk.Poly = pymunk.Poly(self, vertices, radius=radius)
        self.points = [self.local_to_world(v) for v in self.shape.get_vertices()]

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.polygon(screen, Scheme.OBSTACLE, self.points)

    def _get_dimensions(self) -> pymunk.BB:
        """
        Get the bounding box dimensions of the polygonal obstacle.
        Returns (start_x, bottom_y, end_x, top_y).
        """
        start_x, bottom_y = self.position.int_tuple
        end_x, top_y = start_x, bottom_y
        for point in self.points:
            x, y = point.int_tuple
            start_x, bottom_y = min(start_x, x), min(bottom_y, y)
            end_x, top_y = max(end_x, x), max(top_y, y)
        return pymunk.BB(start_x, top_y, end_x, bottom_y)


class TriangularObstacle(PolyObstacle):
    """Represents a triangular obstacle, defined by three vertices."""

    def __init__(
        self,
        base: float,
        height: float,
        position: Vec2,
        velocity: Vec2 = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 0,
        body_type: int = Body.STATIC,
        radius: float = 0,
    ):
        hb, hh = base / 2, height / 2
        self.vertices: List[Vec2] = [(-hb, hh), (hb, hh), (0, -hh)]
        super().__init__(
            self.vertices, position, velocity, field, mass, body_type, radius
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
        position: Vec2,
        dimensions: Vec2,
        orientation: Union[int, Orient] = Orient.HORIZONTAL,
        thickness: int = 3,
        field: Optional[PotentialField] = None,
    ):
        moment = pymunk.moment_for_box(0, dimensions)
        super().__init__(position, field=field, moment=moment, body_type=Body.STATIC)
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
        for segment in self.segments:
            start_pos = Vec2d(*(self.position + segment.a)).int_tuple
            end_pos = Vec2d(*(self.position + segment.b)).int_tuple
            pygame.draw.line(
                screen, Scheme.OBSTACLE, start_pos, end_pos, self.thickness
            )

    def _is_inside(self, coord: Vec2d) -> bool:
        """Check if a coordinate is inside the tunnel."""
        hd = self.dimensions / 2
        return (
            self.position.x - hd.x <= coord.x <= self.position.x + hd.x
            and self.position.y - hd.y <= coord.y <= self.position.y + hd.y
        )

    def _get_dimensions(self) -> pymunk.BB:
        """
        Get the bounding box dimensions of the tunnel.
        Returns (start_x, bottom_y, end_x, top_y).
        """
        hd = self.dimensions / 2
        start = (self.position - hd).int_tuple
        end = (self.position + hd).int_tuple
        return pymunk.BB(*start, *end)


class PointRobot(PointBody):
    """
    Represents a point robot in the simulation. \n
    The robot can move and interact with potential fields and other bodies.
    """

    def __init__(
        self,
        position: Vec2,
        velocity: Vec2 = Vec2d.zero(),
        field: Optional[PotentialField] = None,
        mass: float = 1,
        vmax: float = 10,
        radius: float = 3,
    ):
        super().__init__(Scheme.ROBOT, position, velocity, field, mass, radius=radius)
        self.vmax = vmax  # Maximum horizontal velocity capability

    def step(self) -> None:
        """Ensure robot's horizontal velocity does not exceed the maximum capable limit."""
        if self.velocity.x > self.vmax:
            self.velocity = self.velocity.normalized() * self.vmax


class AttractiveField(PotentialField):
    """
    Represents an radial attractive field in the simulation. \n
    Attracts bodies towards a source with a force proportional to the distance.
    Additionally, a derivative term can be added to the force to stabilize the convergence. \n
    The force field at a point `x` is given by `-k_p * (x - x0)`, where `x0` is the source position.
    For stability, it is written as `-k_v * (x - m)`, where `v` is the body velocity, `m = k_p / k_v * (x0 - x)` and `gamma` is a convergence factor.
    - `k_p`: Proportional constant for the attractive force.
    - `k_v`: Derivative constant for the attractive force.
    - `source`: The point body that exerts the attractive force.
    - `body`: The body that is attracted towards the source, here the point robot.
    - `only_kp`: If True, only the proportional term of the force is considered.
    - `asymptotic_convergence`: If True, the attractive force will converge asymptotically to the source.
    """

    def __init__(
        self,
        k_p: float,
        k_v: float,
        source: Optional[Goal],
        body: Optional[PointRobot],
        only_kp: bool = False,
        asymptotic_convergence: bool = False,
    ):
        self.k_p = k_p
        self.k_v = k_v
        self.source = source
        self.body = body
        self.only_kp = only_kp
        self._asymptotic_convergence = asymptotic_convergence

    def draw(self, screen: pygame.Surface) -> None:
        pass

    def get_potential_field(self, coord: Vec2d) -> float:
        return 0.0
        diff = coord - self.body.position
        return 0.5 * self.k_p * diff.dot(diff)

    def get_force_field(self, coord: Vec2d) -> Vec2d:
        diff = coord - self.source.position
        if self.only_kp:
            return -self.k_p * diff
        m = (self.k_p / self.k_v) * (-diff)
        gamma = 1
        if self._asymptotic_convergence:
            gamma = min(1, self.body.vmax / m.length)
        force = -self.k_v * (self.body.velocity - gamma * m)
        return force

    @property
    def asymptotic_convergence(self) -> bool:
        """
        Convergence mode of the attractive field. \n
        If True, the attractive force will converge asymptotically to the source,
        by enabling the derivative term of the source in the force calculation.
        """
        return self._asymptotic_convergence

    @asymptotic_convergence.setter
    def asymptotic_convergence(self, value: bool) -> None:
        """Set the convergence mode of the attractive field."""
        self._asymptotic_convergence = value


class RepulsiveRadialField(PotentialField):
    """
    Represents a radial repulsive field in the simulation. \n
    Exerts a force on bodies within a certain radius (virtual periphery) around the source.
    The source is modeled as a point body. \n
    The force field at a point `x` is given by `k_r * (1/d - 1/d0) * 1 / d^2`,
    where `d` is the distance between `x` and the source position.
    - `k_r`: Repulsion constant for the repulsive force.
    - `d0`: Virtual periphery radius around the source.
    - `body`: The source body that exerts the repulsive force.
    """

    def __init__(self, k_r: float, d0: float, body: Optional[PointBody] = None):
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


class RepulsiveVirtualPeriphery(PotentialField):
    """
    Represents a virtual periphery around a body in the simulation. \n
    Exerts a repulsive force on bodies within a certain periphery around the source.
    Mimics the shape of the body with a virtual periphery radius. \n
    The force field at a point `x` is given by `k_r * (1/d - 1/d0) * 1 / d^2`,
    where `d` is the distance between `x` and the source position.
    - `k_r`: Repulsion constant for the repulsive force.
    - `d0`: Virtual periphery radius around the source.
    - `body`: The source body that exerts the repulsive force.
    """

    def __init__(self, k_r: float, d0: float, body: PolyObstacle):
        self.k_r = k_r
        self.d0 = d0  # Virtual periphery radius
        self.body = body

    def draw(self, screen: pygame.Surface) -> None:
        points: List[Vec2d] = []
        for vertex in self.body.points:
            direction = (vertex - self.body.position).normalized()
            points.append(vertex + direction * self.d0)
        min_x, max_x = min(p.x for p in points), max(p.x for p in points)
        min_y, max_y = min(p.y for p in points), max(p.y for p in points)
        size = Vec2d(max_x - min_x, max_y - min_y).int_tuple
        surface = pygame.Surface(size, pygame.SRCALPHA)
        shifted_points = [(p.x - min_x, p.y - min_y) for p in points]
        pygame.draw.polygon(surface, Scheme.FIELD, shifted_points)
        screen.blit(surface, (min_x, min_y))

    def get_potential_field(self, coord: Vec2d) -> float:
        d = self._get_distance_to_boundary(coord)
        if d >= self.d0:
            return 0.0
        return 0.5 * self.k_r * (1 / d - 1 / self.d0) ** 2

    def get_force_field(self, coord: Vec2d) -> Vec2d:
        d = self._get_distance_to_boundary(coord)
        if d >= self.d0 or d <= 1e-6:
            return Vec2d.zero()
        direction = (coord - self.body.position).normalized()
        force_mag = self.k_r * (1 / d - 1 / self.d0) * (1 / d**2)
        if math.isnan(force_mag) or math.isnan(direction.length):
            return Vec2d.zero()
        return force_mag * direction

    def _get_distance_to_boundary(self, coord: Vec2d) -> float:
        """Compute the closest distance from `coord` to the obstacle boundary."""

        def _distance_to_segment(p: Vec2d, v1: Vec2d, v2: Vec2d) -> float:
            """Compute the shortest distance from `p` to the segment (v1, v2)."""
            segment = v2 - v1
            ls = segment.get_length_sqrd()
            if ls == 0:
                return (p - v1).length  # `v1` and `v2` are the same point
            # Project `p` onto the line segment
            t = max(0, min(1, ((p - v1).dot(segment) / ls)))
            projection = v1 + t * segment  # Closest point on segment
            return (p - projection).length  # Distance from `p` to projection

        points = self.body.points
        min_d = float("inf")
        for i, v1 in enumerate(points):
            v2 = points[(i + 1) % len(points)]
            min_d = min(min_d, _distance_to_segment(coord, v1, v2))
        return min_d


class TunnelField(PotentialField):
    """
    Represents a tunnel field in the simulation. \n
    Exerts a force on bodies passing through it.
    """

    def __init__(self, strength: float, body: Tunnel):
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
    - `display_size`: Size of the display window in pixels, or `"full"` for fullscreen.
    - `elasticity`: Coefficient of restitution for collisions.
    - `ground_y`: Y-coordinate of the ground level, or `None` to set it near the bottom of the display window.
        The top-left corner is the origin `(0, 0)`, with positive y-axis pointing downwards.
    - `time_step`: Time step for the simulation.
    - `sub_steps`: Number of sub-steps per time step.
        Increase this value for smoother simulations.
    """

    def __init__(
        self,
        display_size: Union[Tuple[Optional[int], Optional[int]], str] = "full",
        elasticity: float = 1.0,
        ground_y: Optional[int] = None,
        time_step: float = 0.1,
        sub_steps: int = 10,
    ):
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
            display_params = {"flags": pygame.FULLSCREEN}
        elif isinstance(self.size, Tuple):
            # Default to display size if any dimension is not provided
            size = [None, None]
            size[0] = self.size[0] or pygame.display.Info().current_w
            size[1] = self.size[1] or pygame.display.Info().current_h
            display_params = {"size": size, "flags": pygame.RESIZABLE}
        self.screen: pygame.Surface = pygame.display.set_mode(**display_params)

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

    def get_bbox(self) -> pymunk.BB:
        """Get the bounding box of the simulation environment."""
        return pymunk.BB(0, self.screen.get_height(), self.screen.get_width(), 0)

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
        self,
        stopping_condition: Optional[Callable[[], bool]] = None,
        framerate: int = 60,
        record: bool = False,
        filename: str = "simulation.mp4",
    ) -> None:
        """
        Start the simulation and render the environment.
        Press `Esc` or close the window to stop the simulation.
        - `stopping_condition`: Optional function that returns a bool. If True, the simulation will stop.
        - `framerate`: Frames per second for rendering the simulation.
        - `record`: If True, the simulation will be recorded as a video.
        - `filename`: Name/Path of the output video file.
        """

        running, completed = True, False
        clock = pygame.time.Clock()
        frames: List[pygame.surfarray.array3d] = []
        while running and not completed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            if stopping_condition is not None and stopping_condition():
                running, completed = False, True

            for func in self.pipeline:
                func()
            self.apply_effects()
            self.screen.fill(Scheme.BACKGROUND)

            # Draw bodies and fields
            for field in self.fields:
                if field.draw_below:
                    field.draw(self.screen)
            for body in self.bodies:
                if body.field is not None and body.field.draw_below:
                    body.field.draw(self.screen)
                body.draw(self.screen)
                if body.field is not None and not body.field.draw_below:
                    body.field.draw(self.screen)
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
                    self.ground_y,
                    self.screen.get_width(),
                    self.screen.get_height() - self.ground_y,
                ),
            )
            pygame.draw.line(
                self.screen,
                Scheme.GROUND,
                (0, self.ground_y),
                (self.screen.get_width(), self.ground_y),
                width=1,
            )

            if record:
                frame = pygame.surfarray.array3d(self.screen)
                frame = pygame.transform.rotate(
                    pygame.surfarray.make_surface(frame), -90
                )
                frame = pygame.transform.flip(frame, True, False)  # Flip horizontally
                frames.append(pygame.surfarray.array3d(frame))

            # Step the simulation
            for _ in range(self.sub_steps):
                self.space.step(self.dt / self.sub_steps)
            pygame.display.flip()  # Update the display
            clock.tick(framerate)
        pygame.quit()

        # Export
        if record and completed and frames:
            clip = moviepy.ImageSequenceClip(frames, fps=framerate)
            clip.write_videofile(filename, codec="libx264", fps=framerate)
            clip.close()
