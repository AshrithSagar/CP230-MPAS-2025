"""
play.py
Play Hamstrung sqaud game
"""

import math
import os
from ast import literal_eval
from enum import IntEnum
from typing import Optional, Tuple

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
from rich.console import Console
from rich.prompt import Prompt

Coord = Tuple[int, int]
console = Console()
prompt = Prompt(console=console)


class HamstrungSquadGame:
    """Hamstrung squad game"""

    class Turn(IntEnum):
        PURSUER = 0
        EVADER = 1

    class Direction(IntEnum):
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

    def __init__(self, evader: Optional[Coord] = None, control_scheme: str = "reduced"):
        self.cell_size = 30
        self.max_grid_size = 20
        self.width = self.height = self.max_grid_size * self.cell_size
        assert control_scheme in ["reduced", "full"]
        self.control_scheme = control_scheme
        self.pursuer: Coord = [1, self.max_grid_size - 1]
        self.pursuer_direction = self.Direction.UP
        self.pursuer_velocity = 2
        self.evader: Coord = (
            [self.max_grid_size - 1, 0]
            if not evader
            else [evader[0], self.max_grid_size - evader[1] - 1]
        )
        self.evader_velocity = 1
        self.game_over = False
        self.clock = pygame.time.Clock()
        self.turn = self.Turn.PURSUER
        self.payoff = 0
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Hamstrung squad game")

    def draw_grid(self):
        """Draw the grid and characters on the screen."""
        # Grid
        self.screen.fill((0, 0, 0))
        for i in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, (255, 255, 255), (i, 0), (i, self.height))
            pygame.draw.line(self.screen, (255, 255, 255), (0, i), (self.width, i))

        # Pursuer
        px, py = self.pursuer
        angle = self.direction_to_angle(self.pursuer_direction)
        pursuer_points = self.get_rotated_triangle_points(px, py, angle)
        pygame.draw.polygon(self.screen, (0, 255, 0), pursuer_points)
        if self.turn == self.Turn.PURSUER:
            pygame.draw.polygon(self.screen, (255, 255, 0), pursuer_points, width=2)

        # Evader
        ex, ey = self.evader
        radius = self.cell_size // 2
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            (ex * self.cell_size + radius, ey * self.cell_size + radius),
            radius,
        )
        if self.turn == self.Turn.EVADER:
            pygame.draw.circle(
                self.screen,
                (255, 255, 0),
                (ex * self.cell_size + radius, ey * self.cell_size + radius),
                radius,
                width=2,
            )

        pygame.display.flip()

    def direction_to_angle(self, direction: Direction) -> float:
        """Convert a direction to an angle in radians."""
        return {
            self.Direction.UP: 0,
            self.Direction.RIGHT: math.pi / 2,
            self.Direction.DOWN: math.pi,
            self.Direction.LEFT: -math.pi / 2,
        }[direction]

    def get_rotated_triangle_points(self, x: int, y: int, angle: float):
        """Returns the points of the triangle rotated based on the direction."""
        base, height = self.cell_size, self.cell_size
        points = [
            (x * self.cell_size, y * self.cell_size - height // 2),
            (x * self.cell_size - base // 2, y * self.cell_size + height // 2),
            (x * self.cell_size + base // 2, y * self.cell_size + height // 2),
        ]
        rotated_points = []
        for point in points:
            rotated_point = self.rotate_point(
                point[0],
                point[1],
                x * self.cell_size,
                y * self.cell_size,
                angle,
            )
            rotated_points.append(rotated_point)
        return rotated_points

    def rotate_point(
        self, px: int, py: int, cx: int, cy: int, angle: float
    ) -> Tuple[int, int]:
        """Rotate a point around a center point by a given angle."""
        s, c = math.sin(angle), math.cos(angle)
        px -= cx
        py -= cy
        new_x = cx + (px * c - py * s)
        new_y = cy + (px * s + py * c)
        return int(new_x), int(new_y)

    def handle_input(self):
        """Handle player input for movement."""
        move_keys = {
            self.Turn.PURSUER: {
                pygame.K_w: self.Direction.UP,
                pygame.K_a: self.Direction.LEFT,
                pygame.K_s: self.Direction.DOWN,
                pygame.K_d: self.Direction.RIGHT,
            },
            self.Turn.EVADER: {
                pygame.K_UP: (0, -1),
                pygame.K_LEFT: (-1, 0),
                pygame.K_DOWN: (0, 1),
                pygame.K_RIGHT: (1, 0),
            },
        }
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                self.game_over = True
                return
            if event.type == pygame.KEYDOWN and event.key in move_keys[self.turn]:
                if self.turn == self.Turn.PURSUER:
                    move_type = move_keys[self.turn][event.key]
                    if self.control_scheme == "reduced":
                        if move_type == self.Direction.UP:
                            dx, dy = self.direction_to_delta(self.pursuer_direction)
                        elif move_type == self.Direction.RIGHT:
                            self.pursuer_direction = (self.pursuer_direction + 1) % 4
                            dx, dy = self.direction_to_delta(self.pursuer_direction)
                        else:
                            continue
                    elif self.control_scheme == "full":
                        if move_type == self.pursuer_direction or (
                            move_type == (self.pursuer_direction + 1) % 4
                        ):
                            dx, dy = self.direction_to_delta(move_type)
                            self.pursuer_direction = move_type
                        else:
                            continue
                    self.pursuer = [
                        self.clamp(self.pursuer[0] + dx * self.pursuer_velocity),
                        self.clamp(self.pursuer[1] + dy * self.pursuer_velocity),
                    ]
                    self.payoff += 1
                else:
                    dx, dy = move_keys[self.turn][event.key]
                    self.evader = [
                        self.clamp(self.evader[0] + dx * self.evader_velocity),
                        self.clamp(self.evader[1] + dy * self.evader_velocity),
                    ]
                break

    def direction_to_delta(self, direction: Direction) -> Tuple[int, int]:
        """Convert a direction to a delta (dx, dy)."""
        return {
            self.Direction.UP: (0, -1),
            self.Direction.RIGHT: (1, 0),
            self.Direction.DOWN: (0, 1),
            self.Direction.LEFT: (-1, 0),
        }[direction]

    def clamp(self, value: int) -> int:
        """Clamp a value between 0 and max_grid_size - 1."""
        return max(0, min(self.max_grid_size - 1, value))

    def is_game_over(self) -> bool:
        """Check if pursuer has caught the evader."""
        return (
            abs(self.pursuer[0] - self.evader[0]) <= 1
            and abs(self.pursuer[1] - self.evader[1]) <= 1
        )

    def play(self):
        """Run the game loop."""
        while not self.game_over:
            self.clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
            self.draw_grid()
            self.handle_input()
            if self.is_game_over():
                console.print(f"[purple]Game over![/] Payoff: {self.payoff}")
                break
            self.turn = (
                self.Turn.EVADER
                if self.turn == self.Turn.PURSUER
                else self.Turn.PURSUER
            )
        pygame.quit()


if __name__ == "__main__":
    evader: Coord = literal_eval(prompt.ask("[green]Evader's starting position[/]"))
    game = HamstrungSquadGame(evader=evader, control_scheme="reduced")
    game.play()
