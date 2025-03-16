"""
play.py
Play Hamstrung sqaud game
"""

from enum import Enum
from typing import Optional, Tuple

import pygame
from rich.prompt import Prompt

Coord = Tuple[int, int]


class HamstrungSquadGame:
    """Hamstrung squad game"""

    class Turn(Enum):
        PURSUER = 0
        EVADER = 1

    def __init__(self, evader: Optional[Coord] = None):
        self.cell_size = 30
        self.max_grid_size = 20
        self.width, self.height = (
            self.max_grid_size * self.cell_size,
            self.max_grid_size * self.cell_size,
        )
        self.pursuer: Coord = [0, self.max_grid_size - 1]
        if not evader:
            evader = (Coord)(self.max_grid_size - 1, 0)
        else:
            evader = (evader[0], self.max_grid_size - evader[1] - 1)
        self.evader = evader
        self.game_over = False
        self.clock = pygame.time.Clock()
        self.turn = self.Turn.PURSUER
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Hamstrung squad game")

    def draw_grid(self):
        """Draw the grid, pursuer and evader on the screen."""
        self.screen.fill((0, 0, 0))
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, (255, 255, 255), (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, (255, 255, 255), (0, y), (self.width, y))
        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (
                self.pursuer[0] * self.cell_size,
                self.pursuer[1] * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
        )
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            (
                self.evader[0] * self.cell_size,
                self.evader[1] * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
        )
        pygame.display.flip()

    def handle_input(self):
        """Handle input for the pursuer and evader."""
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        move_keys = {
            self.Turn.PURSUER: [
                (pygame.K_w, 0, -1),
                (pygame.K_s, 0, 1),
                (pygame.K_a, -1, 0),
                (pygame.K_d, 1, 0),
            ],
            self.Turn.EVADER: [
                (pygame.K_UP, 0, -1),
                (pygame.K_DOWN, 0, 1),
                (pygame.K_LEFT, -1, 0),
                (pygame.K_RIGHT, 1, 0),
            ],
        }
        for key, x, y in move_keys[self.turn]:
            if keys[key]:
                dx, dy = x, y
        if self.turn == self.Turn.PURSUER:
            self.pursuer: Coord = [
                max(0, min(self.max_grid_size - 1, self.pursuer[0] + dx)),
                max(0, min(self.max_grid_size - 1, self.pursuer[1] + dy)),
            ]
        elif self.turn == self.Turn.EVADER:
            self.evader: Coord = [
                max(0, min(self.max_grid_size - 1, self.evader[0] + dx)),
                max(0, min(self.max_grid_size - 1, self.evader[1] + dy)),
            ]

    def is_game_over(self):
        """Check if the game is over."""
        x_diff = abs(self.pursuer[0] - self.evader[0])
        y_diff = abs(self.pursuer[1] - self.evader[1])
        if x_diff <= 1 and y_diff <= 1:
            return True
        return False

    def play(self):
        """Play the game."""
        while not self.game_over:
            self.clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
            self.handle_input()
            self.draw_grid()
            if self.is_game_over():
                print("Game Over! Evader in capture region of pursuer.")
                self.game_over = True
            self.turn = (
                self.Turn.EVADER
                if self.turn == self.Turn.PURSUER
                else self.Turn.PURSUER
            )
        pygame.quit()


if __name__ == "__main__":
    evader: Coord = eval(Prompt.ask("[green]Evader's starting position[/]"))
    game = HamstrungSquadGame(evader=evader)
    game.play()
