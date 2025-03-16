"""
play.py
Play Hamstrung sqaud game
"""

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

    def __init__(self, evader: Optional[Coord] = None):
        self.cell_size = 30
        self.max_grid_size = 20
        self.width = self.height = self.max_grid_size * self.cell_size
        self.pursuer: Coord = [0, self.max_grid_size - 1]
        self.pursuer_direction = [0, 2]
        self.evader: Coord = (
            [self.max_grid_size - 1, 0]
            if not evader
            else [evader[0], self.max_grid_size - evader[1] - 1]
        )
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
        pursuer_x, pursuer_y = self.pursuer
        pursuer_points = [
            (
                pursuer_x * self.cell_size + self.cell_size // 2,
                pursuer_y * self.cell_size,
            ),
            (
                pursuer_x * self.cell_size,
                pursuer_y * self.cell_size + self.cell_size,
            ),
            (
                pursuer_x * self.cell_size + self.cell_size,
                pursuer_y * self.cell_size + self.cell_size,
            ),
        ]
        pygame.draw.polygon(self.screen, (0, 255, 0), pursuer_points)

        # Evader
        evader_x, evader_y = self.evader
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            (
                evader_x * self.cell_size,
                evader_y * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
        )

        # Highlight
        highlight_color = (255, 255, 0)
        highlight_thickness = 5
        if self.turn == self.Turn.PURSUER:
            highlight_x, highlight_y = pursuer_x, pursuer_y
        else:
            highlight_x, highlight_y = evader_x, evader_y
        pygame.draw.rect(
            self.screen,
            highlight_color,
            (
                highlight_x * self.cell_size,
                highlight_y * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
            highlight_thickness,
        )
        pygame.display.flip()

    def handle_input(self):
        """Handle player input for movement."""
        move_keys = {
            self.Turn.PURSUER: {
                pygame.K_w: (0, -2),
                pygame.K_s: (0, 2),
                pygame.K_a: (-2, 0),
                pygame.K_d: (2, 0),
            },
            self.Turn.EVADER: {
                pygame.K_UP: (0, -1),
                pygame.K_DOWN: (0, 1),
                pygame.K_LEFT: (-1, 0),
                pygame.K_RIGHT: (1, 0),
            },
        }
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                self.game_over = True
                return
            if event.type == pygame.KEYDOWN and event.key in move_keys[self.turn]:
                dx, dy = move_keys[self.turn][event.key]
                break
        if self.turn == self.Turn.PURSUER:
            if (dx, dy) in [
                self.pursuer_direction,
                (self.pursuer_direction[1], -self.pursuer_direction[0]),
            ]:
                self.pursuer = [
                    self.clamp(self.pursuer[0] + dx),
                    self.clamp(self.pursuer[1] + dy),
                ]
                self.pursuer_direction = (dx, dy)
                self.payoff += 1
            else:
                self.turn = self.Turn.PURSUER
        else:
            self.evader = [
                self.clamp(self.evader[0] + dx),
                self.clamp(self.evader[1] + dy),
            ]

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
    game = HamstrungSquadGame(evader=evader)
    game.play()
