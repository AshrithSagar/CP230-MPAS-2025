"""
utils.py \n
Utility functions
"""

import random
from collections import deque
from typing import Dict, Generator, List, Tuple

Point = Tuple[int, int]
"""A point (x, y) in the grid"""


class GridMap:
    """Grid map with obstacles and explored areas."""

    def __init__(self, size: int, num_obstacles: int, obs_size: int):
        """
        Initialize grid map with given size and number of obstacles.

        :param size: Size of the grid (N x N)
        :param num_obstacles: Number of block‐shaped obstacles
        :param obs_size: Size of each obstacle (obs_size x obs_size)
        """
        self.N = size
        self.free = [[True] * size for _ in range(size)]
        self.explored = [[False] * size for _ in range(size)]

        # Randomly place block‐shaped obstacles
        for _ in range(num_obstacles):
            x = random.randrange(size - obs_size)
            y = random.randrange(size - obs_size)
            for i in range(obs_size):
                for j in range(obs_size):
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
            q = (x + dx, y + dy)
            if self.in_bounds(q) and self.is_free(q):
                yield q

    def mark_explored(self, robot_pos: Point, sensor_range: int):
        """
        Robot sees in a square of side (2r+1).
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
        The state is represented as a matrix:
        - 0: unknown
        - 1: explored free
        - 2: frontier
        - 3: obstacle

        :return: Tuple of grid state and list of frontier points
        """
        N = self.N
        front = set(self.frontiers())
        state = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if not self.free[i][j]:
                    state[i][j] = 3
                elif (i, j) in front:
                    state[i][j] = 2
                elif self.explored[i][j]:
                    state[i][j] = 1
                else:
                    state[i][j] = 0
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

    def __init__(self, id: int, start: Point, sensor_range: int):
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

    def step(self):
        """Advance the robot by one step along its path."""
        if len(self.path) > 1:
            # Advance by one cell
            self.pos = self.path[1]
            self.path.pop(0)


class Coordinator:
    """Coordinator for multiple robots to explore the grid."""

    def __init__(self, grid: GridMap, robots: List[Robot]):
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

    def assign(self):
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
