"""
utils.py \n
Utility functions
"""

import logging
import random
from collections import deque
from enum import Enum
from typing import Dict, Generator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


Point = Tuple[int, int]
"""A point (x, y) in the grid"""


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    :param seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.debug(f"Random seed: {seed}")


class CellState(Enum):
    """Cell state for the grid cells."""

    UNKNOWN = ("lightgray", 0)
    EXPLORED = ("white", 1)
    FRONTIER = ("yellow", 2)
    OBSTACLE = ("dimgray", 3)

    @staticmethod
    def get_cmap() -> ListedColormap:
        """
        Generate a colormap for matplotlib based on CellState colors.

        :return: ListedColormap object
        """
        return ListedColormap([state.value[0] for state in CellState])


class GridMap:
    """Grid map with obstacles and explored areas."""

    def __init__(self, grid_size: int, num_obstacles: int, obstacle_size: int) -> None:
        """
        Initialize grid map with given size and number of obstacles.

        :param grid_size: Size of the grid (N x N)
        :param num_obstacles: Number of block‐shaped obstacles
        :param obs_size: Size of each obstacle (obs_size x obs_size)
        """
        self.N = grid_size
        self.free = [[True] * grid_size for _ in range(grid_size)]
        self.explored = [[False] * grid_size for _ in range(grid_size)]

        # Randomly place block‐shaped obstacles
        for _ in range(num_obstacles):
            x = random.randrange(grid_size - obstacle_size)
            y = random.randrange(grid_size - obstacle_size)
            for i in range(obstacle_size):
                for j in range(obstacle_size):
                    self.free[x + i][y + j] = False

    def in_bounds(self, p: Point) -> bool:
        """
        Check if a point is within the grid bounds.

        :param p: Point (x, y)
        :return: True if in bounds, False otherwise
        """
        x, y = p
        return 0 <= x < self.N and 0 <= y < self.N

    def is_free(self, p: Point) -> bool:
        """
        Check if a point is free (not an obstacle).

        :param p: Point (x, y)
        :return: True if free, False otherwise
        """
        x, y = p
        return self.free[x][y]

    def neighbors(self, p: Point) -> Generator[Point, None, None]:
        """
        Generate neighboring points (up, down, left, right).

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

    def frontiers(self) -> List[Point]:
        """
        Returns a list of frontier points.
        A frontier point is an explored free cell with at least one unknown neighbor.
        """
        F = []
        for x in range(self.N):
            for y in range(self.N):
                if self.explored[x][y] and self.free[x][y]:
                    # Check for unknown neighbor
                    for q in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                        if self.in_bounds(q) and not self.explored[q[0]][q[1]]:
                            F.append((x, y))
                            break
        return F

    def grid_state(self) -> Tuple[List[List[int]], List[Point]]:
        """
        Returns the state of the grid and the list of frontier points.
        The state is represented as a matrix, as specified by the CellState enum.

        :return: Tuple of grid state and list of frontier points
        """
        N = self.N
        front = set(self.frontiers())
        state = [[CellState.UNKNOWN.value[1]] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if not self.free[i][j]:
                    state[i][j] = CellState.OBSTACLE.value[1]
                elif (i, j) in front:
                    state[i][j] = CellState.FRONTIER.value[1]
                elif self.explored[i][j]:
                    state[i][j] = CellState.EXPLORED.value[1]
                else:
                    state[i][j] = CellState.UNKNOWN.value[1]
        return state, list(front)

    def bfs(self, start: Point, goal: Point) -> List[Point]:
        """
        Return shortest path from start to goal (or [] if unreachable).
        Performs a breadth-first search (BFS) to find the path.

        :param start: Starting point (x, y)
        :param goal: Goal point (x, y)
        :return: List of points representing the path
        """
        if start == goal:
            return [start]
        q = deque([start])
        prev: Dict[Point, Point] = {}
        visited = {start}
        while q:
            u = q.popleft()
            for v in self.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    prev[v] = u
                    if v == goal:
                        # Reconstruct
                        path = [v]
                        while path[-1] != start:
                            path.append(prev[path[-1]])
                        return list(reversed(path))
                    q.append(v)
        return []


class Robot:
    """Robot with a sensor range and a path to follow."""

    def __init__(self, id: int, start: Point, sensor_range: int) -> None:
        """
        Initialize robot with ID, start position, and sensor range.

        :param id: Robot ID
        :param start: Starting position (x, y)
        :param sensor_range: Sensor range (in cells)
        """
        self.id = id
        self.pos = start
        self.sensor_range = sensor_range
        self.path: List[Point] = []

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

    @classmethod
    def from_count(cls, count: int, start: Point, sensor_range: int) -> List["Robot"]:
        """
        Create a list of robots with unique IDs.

        :param count: Number of robots
        :param start: Starting position (x, y)
        :param sensor_range: Sensor range (in cells)
        :return: List of Robot objects
        """
        return [cls(i + 1, start, sensor_range) for i in range(count)]


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
        F = self.grid.frontiers()

        # Initialize utilities if first step
        for f in F:
            self.U.setdefault(f, 1.0)

        # Greedy one‑by‑one
        for r in self.robots:
            best, best_score = None, float("inf")
            for f in F:
                c = r.cost_to(f)
                score = c - self.U[f]
                if score < best_score:
                    best_score, best = score, f
            if best is None:
                continue

            # Plan path
            r.path = self.grid.bfs(r.pos, best)

            # Update utilities of other frontiers
            for t in F:
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

    def setup(self) -> None:
        """Setup the initial state of the scene."""
        # Mark the initial explored area for each robot
        for robot in self.robots:
            self.grid_map.mark_explored(robot.pos, robot.sensor_range)

        # Prepare figure once
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.tight_layout()
        N = self.grid_map.N
        cmap = CellState.get_cmap()
        self.im = self.ax.imshow(np.zeros((N, N)), origin="upper", cmap=cmap)
        self.robot_colors = plt.cm.tab10(np.linspace(0, 1, len(self.robots)))
        self.texts: List[plt.Text] = []  # Utility texts
        self.lines: List[plt.Line2D] = []  # Path lines
        self.dots: List[plt.Line2D] = []  # Robot markers

    def render(
        self, num_iterations: int, delay: float = 0.1, close_after: bool = False
    ) -> None:
        """
        Run the simulation for a specified number of iterations.

        :param num_iterations: Number of iterations to run
        :param delay: Delay between iterations (in seconds)
        :param close_after: If True, close the plot immediately after the simulation ends
        """
        for t in range(1, num_iterations + 1):
            self.coordinator.assign()

            # Update grid image
            state, fronts = self.grid_map.grid_state()
            self.im.set_data(np.array(state))

            # Clear old annotations
            for obj in self.texts + self.lines + self.dots:
                obj.remove()
            self.texts.clear()
            self.lines.clear()
            self.dots.clear()

            # Draw utilities on frontiers
            for x, y in fronts:
                u = self.coordinator.U.get((x, y), 0.0)
                txt = self.ax.text(
                    y,
                    x,
                    f"{u:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )
                self.texts.append(txt)

            for robot, color in zip(self.robots, self.robot_colors):
                # Plot the entire path history
                xs, ys = zip(*[robot.pos] + robot.path)
                self.ax.plot(ys, xs, "-", linewidth=1, color=color, alpha=0.3)

                # Plot the current position
                px, py = robot.pos
                (dot,) = self.ax.plot(py, px, "o", label=f"R{robot.id}", color=color)
                self.dots.append(dot)

            self.ax.set_title(f"Iteration {t}")
            self.ax.legend(loc="upper right", fontsize=8)
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            self.fig.canvas.draw_idle()
            plt.pause(delay)

            # Each robot moves one step and re‑explores
            for robot in self.robots:
                robot.step()
                self.grid_map.mark_explored(robot.pos, robot.sensor_range)

            logger.info(
                f"Step {t}:\n"
                f"  positions:  "
                + ", ".join(f"Robot {r.id}: {r.pos}" for r in self.robots)
            )

        plt.ioff()
        if not close_after:
            plt.show()
        plt.close(self.fig)
        logger.debug("Simulation complete.")
