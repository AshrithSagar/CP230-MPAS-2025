"""
brute_force.py
Brute-force agent for the Hamstrung squad game
"""

from collections import deque

import numpy as np
from hamstrung_squad import HamstrungSquadEnv


class BruteForceAgent:
    def __init__(self, env: HamstrungSquadEnv, max_payoff: int = 10):
        self.env = env
        self.max_payoff = max_payoff

    def train(self) -> None:
        gs = self.env.grid_size
        self.payoff_table = np.full((gs, gs), np.nan)
        for ex, ey in np.ndindex(gs, gs):
            env = self.env.__class__(grid_size=gs)
            env.reset(options={"evader_pos": (ex, ey)})
            payoff = self._optimal_playout(env)
            if payoff <= self.max_payoff:
                self.payoff_table[ex, ey] = payoff
        self._show_payoff_table()

    def _optimal_playout(self, env: HamstrungSquadEnv) -> int:
        """Computes the optimal number of steps to capture the evader."""
        obs = env._get_obs()
        queue = deque([(obs, 0)])
        visited = set()
        visited.add(obs)
        while queue:
            (obs, payoff) = queue.popleft()
            if np.linalg.norm(np.array(obs[0:2]) - np.array(obs[2:4])) <= 1.5:
                return payoff
            if payoff >= self.max_payoff:
                continue
            for action in np.ndindex(2, 4):
                obs, _, _, _, _ = env.step(action, simulate=True)
                if obs not in visited:
                    visited.add(obs)
                    queue.append((obs, payoff + 1))
        return np.inf

    def _show_payoff_table(self):
        print("Payoff table:")
        for row in self.payoff_table:
            print(" ".join(f"{x:3.0f}" if not np.isnan(x) else "  -" for x in row))
