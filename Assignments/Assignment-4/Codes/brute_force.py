"""
brute_force.py
Brute-force agent for the Hamstrung squad game
"""

import timeit
from collections import deque
from typing import Any, Dict

import numpy as np
from hamstrung_squad import HamstrungSquadEnv
from tqdm import tqdm


class BruteForceAgent:
    def __init__(self, env: HamstrungSquadEnv, max_payoff: int = 10):
        self.env = env
        self.max_payoff = max_payoff
        self.grid_size = env.grid_size
        self.payoff_table = np.full((self.grid_size, self.grid_size), np.inf)

    def train(self, timed: bool = True, verbose: bool = True) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        queue = deque()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Initialize terminal states (capture positions)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if np.linalg.norm([x - (self.grid_size - 1), y - 0]) <= 1.5:
                    self.payoff_table[x, y] = 0
                    queue.append((x, y))

        start_time = timeit.default_timer() if timed else None
        pbar = tqdm(desc="Training", total=self.grid_size**2, leave=False)
        while queue:
            ex, ey = queue.popleft()
            evader_payoff = self.payoff_table[ex, ey]

            for evader_move in directions:
                new_ex, new_ey = ex + evader_move[0], ey + evader_move[1]
                if not (0 <= new_ex < self.grid_size and 0 <= new_ey < self.grid_size):
                    continue  # Skip out-of-bounds moves

                # Pursuer minimizes the payoff
                min_payoff = float("inf")
                for pursuer_action in range(2):
                    new_pursuer_pos, _ = self._simulate_pursuer_move(
                        (self.grid_size - 1, 0), 0, pursuer_action
                    )

                    # If capture happens
                    dx = new_pursuer_pos[0] - new_ex
                    dy = new_pursuer_pos[1] - new_ey
                    if np.linalg.norm([dx, dy]) <= 1.5:
                        min_payoff = 1
                    else:
                        min_payoff = min(min_payoff, evader_payoff + 1)

                # Update the evader's best strategy
                if self.payoff_table[new_ex, new_ey] > min_payoff:
                    self.payoff_table[new_ex, new_ey] = min_payoff
                    queue.append((new_ex, new_ey))
        pbar.close()
        if timed:
            elapsed = timeit.default_timer() - start_time
            info["time"] = elapsed
        if verbose:
            print("Training completed:")
            if timed:
                print(f" Time: {elapsed:.3f}s")
            self._show_payoff_table()
        return info

    def _simulate_pursuer_move(self, pursuer_pos, pursuer_dir, action):
        """Simulates the pursuer's movement without modifying the environment."""
        if action == 1:  # Turn right
            pursuer_dir = (pursuer_dir + 1) % 4
        pursuer_delta = np.array([(-2, 0), (0, 2), (2, 0), (0, -2)][pursuer_dir])
        new_pursuer_pos = np.clip(
            np.array(pursuer_pos) + pursuer_delta, 0, self.grid_size - 1
        )
        return tuple(new_pursuer_pos), pursuer_dir

    def _show_payoff_table(self):
        print("Payoff table:")
        for row in self.payoff_table:
            print(" ".join(f"{x:3.0f}" if x != np.inf else "  -" for x in row))
