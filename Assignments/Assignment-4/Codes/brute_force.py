"""
brute_force.py
Brute-force agent for the Hamstrung squad game
"""

import numpy as np
from hamstrung_squad import HamstrungSquadEnv
from numpy.typing import NDArray


class BruteForceAgent:
    def __init__(self, env: HamstrungSquadEnv, max_payoff: int = 10):
        self.env = env
        self.max_payoff = max_payoff
        self.payoff_table = self.compute_payoff_table()

    def compute_payoff_table(self) -> NDArray:
        gs = self.env.grid_size
        payoff_table = np.full((gs, gs), self.max_payoff, dtype=int)
        for ex, ey in np.ndindex(gs, gs):
            env = HamstrungSquadEnv(grid_size=gs)
            env.reset(options={"evader": (ex, ey)})
            payoff = self.optimal_playout(env)
            payoff_table[ex, ey] = payoff
        return payoff_table

    def optimal_playout(self, env: HamstrungSquadEnv) -> int:
        visited = set()
        queue = [(env.pursuer, env.evader, env.pursuer_direction, 0)]
        while queue:
            pursuer, evader, pursuer_dir, steps = queue.pop(0)
            if steps >= self.max_payoff:
                continue
            if np.linalg.norm(np.array(pursuer) - np.array(evader)) <= 1.5:
                return steps
            if (pursuer, evader, pursuer_dir) in visited:
                continue
            visited.add((pursuer, evader, pursuer_dir))
            best_worst_case = self.max_payoff
            for pursuer_action in range(2):  # Forward, Turn Right
                worst_case = 0
                for evader_action in range(4):  # Up, Right, Down, Left
                    next_obs, _, _, _, _ = env.step(
                        (pursuer_action, evader_action), simulate=True
                    )
                    pursuer, evader, pursuer_dir = next_obs[:3]
                    queue.append((pursuer, evader, pursuer_dir, steps + 1))
                    worst_case = max(worst_case, steps + 1)
                best_worst_case = min(best_worst_case, worst_case)
        return best_worst_case

    def show_payoff_table(self):
        print("Payoff table:")
        for row in self.payoff_table:
            print(" ".join(f"{x:2d}" for x in row))
