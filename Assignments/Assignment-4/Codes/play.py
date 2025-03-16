"""
play.py
Play Hamstrung sqaud game
"""

import os
from ast import literal_eval
from enum import IntEnum
from typing import Callable, List, Optional, Tuple

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import numpy as np
import pygame
from rich.console import Console
from rich.prompt import Prompt

Coord = Tuple[int, int]
console = Console()


class HamstrungSquadGame:
    """Hamstrung squad game"""

    class Turn(IntEnum):
        PURSUER = 0
        EVADER = 1

        def __next__(self):
            return self.__class__((self + 1) % 2)

    class Direction(IntEnum):
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

        def next(self):
            return self.__class__((self + 1) % 4)

        def to_delta(self) -> Coord:
            return {
                self.UP: (0, -1),
                self.RIGHT: (1, 0),
                self.DOWN: (0, 1),
                self.LEFT: (-1, 0),
            }[self]

    def __init__(
        self,
        evader: Optional[Coord] = None,
        control_scheme: Optional[str] = None,
        cell_size: int = 30,
        max_grid_size: int = 20,
    ):
        self.cell_size = cell_size
        self.max_grid_size = max_grid_size
        self.width = self.height = self.max_grid_size * self.cell_size
        if control_scheme is None:
            control_scheme = Prompt(console=console).ask(
                "[green]Control scheme[/]",
                choices=["reduced", "full"],
                default="reduced",
            )
        assert control_scheme in ["reduced", "full"]
        self.control_scheme = control_scheme
        self.pursuer: Coord = [1, self.max_grid_size - 1]
        self.pursuer_direction = self.Direction.UP
        self.pursuer_velocity: int = 2
        if evader is None:
            evader = Prompt(console=console).ask("[green]Evader's starting position[/]")
            evader = literal_eval(evader)
            self.evader: Coord = [evader[0] + 1, self.max_grid_size - evader[1] - 1]
        self.evader_velocity: int = 1
        self.payoff: int = 0
        self.game_over: bool = False
        self.turn = self.Turn.PURSUER
        self.clock = pygame.time.Clock()
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Hamstrung squad game")

    def handle_update(self):
        """Draw the grid and characters on the screen."""
        cs = self.cell_size

        # Grid
        self.screen.fill((0, 0, 0))
        for i in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, (255, 255, 255), (i, 0), (i, self.height))
            pygame.draw.line(self.screen, (255, 255, 255), (0, i), (self.width, i))

        # Pursuer
        (px, py), b, h = self.pursuer, cs, cs
        triangle: List[Coord]
        if self.pursuer_direction == self.Direction.UP:
            triangle = [(0, -h // 2), (-b // 2, h // 2), (b // 2, h // 2)]
        elif self.pursuer_direction == self.Direction.RIGHT:
            triangle = [(h // 2, 0), (-h // 2, -b // 2), (-h // 2, b // 2)]
        elif self.pursuer_direction == self.Direction.DOWN:
            triangle = [(0, h // 2), (-b // 2, -h // 2), (b // 2, -h // 2)]
        elif self.pursuer_direction == self.Direction.LEFT:
            triangle = [(-h // 2, 0), (h // 2, -b // 2), (h // 2, b // 2)]
        pursuer: List[Coord] = [(px * cs + x, py * cs + y) for x, y in triangle]
        pygame.draw.polygon(self.screen, (0, 255, 0), pursuer)
        if self.turn == self.Turn.PURSUER:
            pygame.draw.polygon(self.screen, (255, 255, 0), pursuer, width=2)

        # Evader
        (ex, ey), r = self.evader, cs // 2
        center: Coord = ex * cs, ey * cs
        pygame.draw.circle(self.screen, (255, 0, 0), center, r)
        if self.turn == self.Turn.EVADER:
            pygame.draw.circle(self.screen, (255, 255, 0), center, r, width=2)

        pygame.display.flip()

    def handle_input(self):
        """Handle player input for movement."""
        keys = {
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
        clip: Callable[[int], int] = lambda x: np.clip(x, 0, self.max_grid_size - 1)
        move: Callable[[Coord, Coord, int], Coord]
        move = lambda p, d, v: [clip(p[0] + d[0] * v), clip(p[1] + d[1] * v)]
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                self.game_over = True
                return
            if event.type == pygame.KEYDOWN and event.key in keys[self.turn]:
                if self.turn == self.Turn.PURSUER:
                    move_type = self.Direction(keys[self.turn][event.key])
                    direction = self.pursuer_direction
                    if self.control_scheme == "reduced":
                        if move_type == self.Direction.UP:
                            pass
                        elif move_type == self.Direction.RIGHT:
                            self.pursuer_direction = direction.next()
                        else:
                            continue
                    elif self.control_scheme == "full":
                        if move_type in [direction, direction.next()]:
                            self.pursuer_direction = move_type
                        else:
                            continue
                    delta = self.pursuer_direction.to_delta()
                    self.pursuer = move(self.pursuer, delta, self.pursuer_velocity)
                    self.payoff += 1
                elif self.turn == self.Turn.EVADER:
                    delta = keys[self.turn][event.key]
                    self.evader = move(self.evader, delta, self.evader_velocity)
                break

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
            self.handle_update()
            self.handle_input()
            if self.is_game_over():
                console.print(f"[purple]Game over![/] Payoff: {self.payoff}")
                break
            self.turn = next(self.turn)
        pygame.quit()


if __name__ == "__main__":
    HamstrungSquadGame().play()
