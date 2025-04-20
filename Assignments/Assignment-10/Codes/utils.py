"""
utils.py \n
Utility functions
"""

import logging
import math
import random
from collections import deque
from enum import Enum
from operator import itemgetter
from typing import Dict, Generator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


Point = Tuple[int, int]
"""A point (x, y) in the grid"""


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility.

    :param seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.debug(f"Random seed: {seed}")


class CellState(Enum):
    """
    Cell state for the grid cells.
    Tuple of (index, color, alpha).
    """

    UNKNOWN = (0, "lightgray", 1.0)
    EXPLORED = (1, "blue", 0.1)
    FRONTIER = (2, "magenta", 0.4)
    OBSTACLE = (3, "dimgray", 1.0)

    @staticmethod
    def get_cmap() -> ListedColormap:
        """
        Generate a colormap for matplotlib based on CellState colors.

        :return: ListedColormap object
        """
        return ListedColormap([state.value[1] for state in CellState])


class ObstacleShape(Enum):
    """
    Supported shapes for the obstacles.
    Each shape has some associated parameters.
    """

    BLOCK = 0
    """
    Rectangular block, with area (s1 * s2).

    :params s1: Block width
    :params s2: Block height
    """

    LSHAPE = 1
    """
    L-shaped, with area (s1 * s2 + s3 * s4).
    Divided into three regions, with areas
    (s1 * (s2 - s3)), (s1 * s3), (s3 * s4).

    :params s1: Vertical bar width
    :params s2: Vertical bar length
    :params s3: Horizontal bar width
    :params s4: Horizontal bar length
    """

    TSHAPE = 2
    """
    T-shaped, with area (s1 * s2 + s3 * s4). \n
    By default, the vertical bar is centered on the horizontal bar,
    i.e., (2 * s5 + s3) = s2 => s5 = (s2 - s3) / 2.
    Divided into four regions, with areas
    (s1 * s5), (s1 * s3), (s1 * (s2 - s3 - s5)), (s3 * s4).

    :params s1: Horizontal bar width
    :params s2: Horizontal bar length
    :params s3: Vertical bar width
    :params s4: Vertical bar length
    :params s5: Horizontal junction distance on horizontal bar
    """


class ObstacleGenerator:
    """Generate obstacles with different shapes."""

    def __init__(
        self,
        shape: ObstacleShape = ObstacleShape.BLOCK,
        occupancy: int = 100,
        **kwargs,
    ) -> None:
        """
        Initialize an obstacle with a given shape and occupancy.

        :param shape: Shape of the obstacle; default BLOCK
        :param occupancy: Number of cells occupied by the obstacle; default 100
        :param kwargs: Additional parameters for the shape
        """
        assert isinstance(shape, ObstacleShape), "Invalid shape"
        assert isinstance(occupancy, int) and occupancy > 0, "Invalid occupancy"
        self.shape = shape
        self.occupancy = occupancy
        self.params = self._get_shape_params(**kwargs)
        self.body = self._generate_body()

    def _get_shape_params(self, **kwargs) -> Dict[str, int]:
        """
        Get the parameters for the shape based on occupancy and additional kwargs.
        The parameters are set to default values if not provided.

        :param kwargs: Additional parameters for the shape
        :return: Dictionary of parameters for the shape
        """
        if self.shape == ObstacleShape.BLOCK:
            # Defaults to a square block
            _s = round(math.sqrt(self.occupancy))
            s1: int = kwargs.get("s1", _s)
            s2: int = kwargs.get("s2", int(self.occupancy // s1))
            return {"s1": s1, "s2": s2}

        elif self.shape == ObstacleShape.LSHAPE:
            # Defaults to 3 square blocks
            _s = round(math.sqrt(self.occupancy / 3))
            s1: int = kwargs.get("s1", _s)
            s2: int = kwargs.get("s2", 2 * _s)
            s3: int = kwargs.get("s3", _s)
            s4: int = kwargs.get("s4", _s)
            return {"s1": s1, "s2": s2, "s3": s3, "s4": s4}

        elif self.shape == ObstacleShape.TSHAPE:
            # Defaults to 4 square blocks
            _s = round(math.sqrt(self.occupancy / 4))
            s1: int = kwargs.get("s1", _s)
            s2: int = kwargs.get("s2", 3 * _s)
            s3: int = kwargs.get("s3", _s)
            s4: int = kwargs.get("s4", _s)
            s5: int = kwargs.get("s5", _s)
            return {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}

    def _generate_body(self) -> List[List[int]]:
        """Generate the obstacle body based on the shape and parameters."""

        if self.shape == ObstacleShape.BLOCK:
            s1, s2 = itemgetter("s1", "s2")(self.params)
            body = [[1] * s1 for _ in range(s2)]

        elif self.shape == ObstacleShape.LSHAPE:
            s1, s2, s3, s4 = itemgetter("s1", "s2", "s3", "s4")(self.params)
            height, width = s2, s1 + s4
            body = [[0] * width for _ in range(height)]
            for i in range(s2):  # Vertical bar
                for j in range(s1):
                    body[i][j] = 1
            for i in range(s3):  # Horizontal bar
                for j in range(s4):
                    body[s2 - s3 + i][s1 + j] = 1

        elif self.shape == ObstacleShape.TSHAPE:
            s1, s2, s3, s4, s5 = itemgetter("s1", "s2", "s3", "s4", "s5")(self.params)
            height, width = s1 + s4, s2
            body = [[0] * width for _ in range(height)]
            for i in range(s1):  # Horizontal bar
                for j in range(s2):
                    body[i][j] = 1
            for i in range(s4):  # Vertical bar
                for j in range(s3):
                    body[s1 + i][s5 + j] = 1

        return body

    def get_cells(self) -> List[Point]:
        """
        Get the coordinates of the occupied cells in the obstacle body.

        :return: List of tuples representing the occupied cells
        """
        cells = []
        for i, row in enumerate(self.body):
            for j, cell in enumerate(row):
                if cell == 1:
                    cells.append((i, j))
        return cells

    def get_bounding_box(self) -> Tuple[int, int]:
        """
        Get the bounding box of the obstacle body.

        :return: Tuple of (height, width) of the bounding box
        """
        return len(self.body), len(self.body[0])

    def show(self):
        """Helper to plot the obstacle body separately."""
        plt.imshow(np.array(self.body), cmap="Greys")


class GridMap:
    """Grid map with obstacles and explored areas."""

    def __init__(
        self,
        grid_size: int = 40,
        num_obstacles: int = 6,
        obstacle_occupancy: int = 100,
        obstacle_min_separation: int = 1,
    ) -> None:
        """
        Initialize grid map with given size and number of obstacles.

        :param grid_size: Size of the grid (N x N); default 40
        :param num_obstacles: Number of block‐shaped obstacles; default 6
        :param obstacle_occupancy: Number of cells occupied by each obstacle; default 100
        :param obstacle_min_separation: Minimum separation between obstacles; default 1
        """
        self.grid_size = grid_size
        self.free = [[True] * grid_size for _ in range(grid_size)]
        self.explored = [[False] * grid_size for _ in range(grid_size)]
        self.place_obstacles(num_obstacles, obstacle_occupancy, obstacle_min_separation)

    def in_bounds(self, p: Point) -> bool:
        """
        Check if a point is within the grid bounds.

        :param p: Point (x, y)
        :return: True if in bounds, False otherwise
        """
        x, y = p
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def is_free(self, p: Point) -> bool:
        """
        Check if a point is free (not an obstacle).

        :param p: Point (x, y)
        :return: True if free, False otherwise
        """
        x, y = p
        return self.free[x][y]

    def well_separated(self, p: Point, sep: int = 0):
        """
        Check if a point is separated from others by a given distance.

        :param p: Point (x, y)
        :param sep: Separation distance; default 0
        :return: True if well separated, False otherwise
        """
        x, y = p
        return all(
            self.in_bounds((x + dx, y + dy)) and self.is_free((x + dx, y + dy))
            for dx in range(-sep, sep + 1)
            for dy in range(-sep, sep + 1)
        )

    def place_obstacles(
        self, num_obstacles: int, occupancy: int, min_separation: int
    ) -> None:
        """Randomly place obstacles while ensuring space availability and separation."""

        max_attempts_per_obstacle = 10  # Prevent infinite loops
        msep = min_separation

        # Estimate required space
        required_cells = num_obstacles * occupancy
        effective_grid_cells = sum(row.count(True) for row in self.free)

        if required_cells > effective_grid_cells:
            raise ValueError(
                f"Insufficient space: Need {required_cells} cells, "
                f"but only {effective_grid_cells} free."
            )

        placed_count = 0
        failed_shapes = []

        while placed_count < num_obstacles:
            shape = random.choice(list(ObstacleShape))  # Random shape
            obstacle = ObstacleGenerator(shape=shape, occupancy=occupancy)
            cells, bb = obstacle.get_cells(), obstacle.get_bounding_box()

            success = False
            for _ in range(max_attempts_per_obstacle):
                x = random.randint(0, self.grid_size - bb[0])
                y = random.randint(0, self.grid_size - bb[1])

                if all(self.well_separated((x + i, y + j), msep) for i, j in cells):
                    for i, j in cells:
                        self.free[x + i][y + j] = False
                    success = True
                    break

            if success:
                placed_count += 1
            else:
                failed_shapes.append(shape.name)

        if placed_count < num_obstacles:
            logger.warning(
                f"Warning: Only placed {placed_count}/{num_obstacles} obstacles."
            )
            if failed_shapes:
                logger.warning(f"Failed shapes: {set(failed_shapes)}")

    def get_free_neighbours(self, p: Point) -> Generator[Point, None, None]:
        """
        Generate adjacent points of a given point that are free.

        :param p: Point (x, y)
        :return: Generator of neighboring points
        """
        x, y = p
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            q: Point = (x + dx, y + dy)
            if self.in_bounds(q) and self.is_free(q):
                yield q

    def mark_explored(self, robot_pos: Point, sensor_range: int) -> None:
        """
        Robot sees in a square of side (2 * sensor_range + 1).
        Mark all cells in the square as explored.

        :param robot_pos: Robot position (x, y)
        :param sensor_range: Sensor range (in cells)
        """
        rx, ry = robot_pos
        for dx in range(-sensor_range, sensor_range + 1):
            for dy in range(-sensor_range, sensor_range + 1):
                q = (rx + dx, ry + dy)
                if self.in_bounds(q):
                    self.explored[q[0]][q[1]] = True

    def get_frontiers(self) -> List[Point]:
        """
        Returns a list of frontier points.
        A frontier point is an explored free cell with at least one unknown neighbor.
        """
        frontiers = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.explored[x][y] and self.free[x][y]:
                    # Check for unknown neighbor
                    for q in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                        if self.in_bounds(q) and not self.explored[q[0]][q[1]]:
                            frontiers.append((x, y))
                            break
        return frontiers

    def get_grid_state(self) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Returns the state of the grid and the alpha values for visualization.
        The state is represented as a matrix, as specified by the CellState enum.
        The alpha values are a 2D array of floats, with the same shape as the state.

        :return: Tuple of 2D list representing the grid state, and 2D list of alpha values
        """
        N = self.grid_size
        frontiers = set(self.get_frontiers())
        state = [[CellState.UNKNOWN.value[0]] * N for _ in range(N)]
        alpha = [[CellState.UNKNOWN.value[2]] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if not self.free[i][j]:
                    state[i][j] = CellState.OBSTACLE.value[0]
                    alpha[i][j] = CellState.OBSTACLE.value[2]
                elif (i, j) in frontiers:
                    state[i][j] = CellState.FRONTIER.value[0]
                    alpha[i][j] = CellState.FRONTIER.value[2]
                elif self.explored[i][j]:
                    state[i][j] = CellState.EXPLORED.value[0]
                    alpha[i][j] = CellState.EXPLORED.value[2]
                else:
                    state[i][j] = CellState.UNKNOWN.value[0]
                    alpha[i][j] = CellState.UNKNOWN.value[2]
        return state, alpha

    def get_shortest_path(self, start: Point, goal: Point) -> List[Point]:
        """
        Return shortest path from start to goal (or [] if unreachable).
        Performs a breadth-first search (BFS) to find the path.

        :param start: Starting point (x, y)
        :param goal: Goal point (x, y)
        :return: List of points representing the path
        """
        if start == goal:
            return [start]
        queue = deque([start])
        prev: Dict[Point, Point] = {}
        visited = {start}
        while queue:
            node = queue.popleft()
            for nbr in self.get_free_neighbours(node):
                if nbr not in visited:
                    visited.add(nbr)
                    prev[nbr] = node
                    if nbr == goal:
                        # Reconstruct
                        path = [nbr]
                        while path[-1] != start:
                            path.append(prev[path[-1]])
                        return list(reversed(path))
                    queue.append(nbr)
        return []


class Robot:
    """Robot with a sensor range and a path to follow."""

    def __init__(
        self,
        id: Optional[int] = None,
        start: Point = (0, 0),
        sensor_range: int = 6,
        color: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Initialize robot with ID, start position, and sensor range.

        :param id: Robot ID. If not provided, a random ID is generated.
        :param start: Starting position (x, y); default (0, 0)
        :param sensor_range: Sensor range (in cells); default 6
        """
        if id is None:
            id = random.randint(1, 1000)
        self.id = id
        self.pos = start
        self.sensor_range = sensor_range
        if color is None:
            color = np.random.rand(3)
        self.color = color
        self.path: List[Point] = []
        self.history: List[Point] = [start]

    def __str__(self) -> str:
        return f"Robot {self.id}"

    def cost_to(self, pt: Point) -> int:
        """
        Calculate cost to reach a point.
        The heuristic for the cost is the Manhattan distance.

        :param pt: Target point (x, y)
        :return: Cost (in cells)
        """
        return abs(self.pos[0] - pt[0]) + abs(self.pos[1] - pt[1])

    def step(self) -> None:
        """Advance the robot by one step along its path."""
        if len(self.path) > 1:
            # Advance by one cell
            self.pos = self.path[1]
            self.path.pop(0)
            self.history.append(self.pos)

    @classmethod
    def from_count(
        cls, count: int = 2, start: Point = (0, 0), sensor_range: int = 6
    ) -> List["Robot"]:
        """
        Create a list of robots with unique IDs and random colors.
        All robots start at the same position and have the same sensor range.

        :param count: Number of robots; default 2
        :param start: Starting position (x, y); default (0, 0)
        :param sensor_range: Sensor range (in cells); default 6
        :return: List of Robot objects
        """
        colors = plt.cm.tab10(np.linspace(0, 1, count))
        return [cls(i + 1, start, sensor_range, tuple(colors[i])) for i in range(count)]


class Coordinator:
    """Coordinator for multiple robots to explore the grid."""

    def __init__(self, grid: GridMap, robots: List[Robot]) -> None:
        """
        Initialize the coordinator with a grid and a list of robots.
        The coordinator manages the assignment of frontiers to robots.

        :param grid: GridMap object
        :param robots: List of Robot objects
        """
        self.grid = grid
        self.robots = robots

        # Uniform initial utility
        self.U: Dict[Point, float] = {}

    def P(self, d: int) -> float:
        """
        Utility function for the distance to a frontier.
        The utility decreases linearly with distance.

        :param d: Distance to the frontier
        :return: Utility value (0.0 to 1.0)
        """
        # Simple linear dropoff
        R = self.robots[0].sensor_range
        return max(0.0, 1.0 - d / float(R))

    def assign(self) -> None:
        """
        Assign frontiers to robots based on their costs and utilities.
        Each robot selects the best frontier to explore next.
        The assignment is greedy and one-by-one.
        """
        frontiers = self.grid.get_frontiers()

        # Initialize utilities if first step
        for frontier in frontiers:
            self.U.setdefault(frontier, 1.0)

        # Greedy one‑by‑one
        for robot in self.robots:
            best, best_score = None, float("inf")
            for frontier in frontiers:
                cost = robot.cost_to(frontier)
                score = cost - self.U[frontier]
                if score < best_score:
                    best_score, best = score, frontier
            if best is None:
                continue

            # Plan path
            robot.path = self.grid.get_shortest_path(robot.pos, best)

            # Update utilities of other frontiers
            for t in frontiers:
                d = abs(best[0] - t[0]) + abs(best[1] - t[1])
                self.U[t] -= self.P(d)
                self.U[t] = max(self.U[t], 0.0)


class Scene:
    """Scene class to manage the setup and simulation of the exploration process."""

    def __init__(self, grid_map: GridMap, robots: List[Robot]) -> None:
        """
        Initialize the Scene with a grid map, robots, and a coordinator.

        :param grid_map: Instance of GridMap
        :param robots: List of Robot instances
        """
        self.grid_map = grid_map
        self.robots = robots
        self.coordinator = Coordinator(grid_map, robots)

    def setup(self) -> "Scene":
        """
        Setup the initial state of the scene.
        This includes marking the initial explored area for each robot and preparing the plot.

        :return: The Scene instance
        """
        # Mark the initial explored area for each robot
        for robot in self.robots:
            self.grid_map.mark_explored(robot.pos, robot.sensor_range)

        # Prepare initial plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        plt.tight_layout()
        plt.xlim(-1, self.grid_map.grid_size)
        plt.ylim(self.grid_map.grid_size, -1)
        state, alpha = self.grid_map.get_grid_state()
        self.grid_image = self.ax.imshow(
            state, origin="upper", alpha=alpha, cmap=CellState.get_cmap(), zorder=1
        )
        return self

    def render(
        self,
        num_iterations: int = 100,
        delay_interval: float = 0.1,
        close_after: bool = False,
        record: bool = False,
        fps: int = 24,
        dpi: int = 100,
        save_file: str = "simulation.mp4",
    ) -> "Scene":
        """
        Run the simulation for a specified number of iterations.

        :param num_iterations: Number of iterations to run; default 100
        :param delay_interval: Delay between iterations (in seconds) to ensure plot updates; default 0.1
        :param close_after: If True, close the plot immediately after the simulation ends; default False
        :param record: If True, record the simulation as a video; default False
        :param fps: Frames per second for the video; default 24
        :param dpi: Dots per inch for the video; default 100
        :param save_file: Filename to save the video; default "simulation.mp4"
        :return: The Scene instance
        """
        texts: List[plt.Text] = []  # Utility texts
        lines: List[plt.Line2D] = []  # Path lines
        dots: List[plt.Line2D] = []  # Robot markers
        circles: List[Circle] = []  # Sensor range circles
        obstacles: List[plt.Rectangle] = []  # Obstacles

        writer = None
        if record:
            writer = FFMpegWriter(fps=int(fps), metadata={"title": "Simulation"})
            writer.setup(self.fig, save_file, dpi=dpi)

        for t in range(1, num_iterations + 1):
            self.coordinator.assign()

            # Update grid image
            state, alpha = self.grid_map.get_grid_state()
            self.grid_image.set_data(state)
            self.grid_image.set_alpha(alpha)
            self.grid_image.autoscale()
            self.grid_image.set_zorder(1)

            # Clear old annotations
            objs: List[List[plt.Artist]] = [texts, lines, dots, circles, obstacles]
            for obj in objs:
                for el in obj:
                    el.remove()
                obj.clear()

            # Draw obstacles
            for i in range(self.grid_map.grid_size):
                for j in range(self.grid_map.grid_size):
                    if not self.grid_map.free[i][j]:
                        rect = plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, color="dimgray", zorder=5
                        )
                        self.ax.add_patch(rect)
                        obstacles.append(rect)

            # Draw utilities on frontiers
            frontiers = self.grid_map.get_frontiers()
            for frontier in frontiers:
                x, y = frontier
                u = self.coordinator.U.get(frontier, 0.0)
                txt = self.ax.text(
                    *(y, x),
                    f"{u:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                    zorder=3,
                )
                texts.append(txt)

            for robot in self.robots:
                # Plot the path history
                xs, ys = zip(*robot.history)
                (line,) = self.ax.plot(
                    *(ys, xs), "-", linewidth=1, color=robot.color, alpha=0.5, zorder=2
                )
                lines.append(line)

                # Plot the current position
                px, py = robot.pos
                (dot,) = self.ax.plot(
                    py, px, "o", label=robot, color=robot.color, zorder=4
                )
                dots.append(dot)

                # Add sensor range visualization
                sensor_circle = Circle(
                    (py, px),
                    radius=robot.sensor_range,
                    edgecolor="yellow",
                    facecolor="yellow",
                    alpha=0.5,
                    zorder=1.5,
                )
                self.ax.add_patch(sensor_circle)
                circles.append(sensor_circle)

            self.ax.set_title(f"Iteration {t}")
            self.ax.legend(loc="best", fontsize=8)
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            self.fig.canvas.draw_idle()
            if record:
                writer.grab_frame()
            else:
                plt.pause(delay_interval)

            # Each robot moves one step and re‑explores
            for robot in self.robots:
                robot.step()
                self.grid_map.mark_explored(robot.pos, robot.sensor_range)

            logger.info(
                f"Iteration {t}:\n  Positions:  "
                + ", ".join(f"{r}: {r.pos}" for r in self.robots)
            )

        if record:
            writer.finish()
        plt.ioff()
        if not close_after:
            plt.show()
        plt.close(self.fig)
        logger.debug("Simulation complete.")
        return self
