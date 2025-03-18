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

    def train(self, timed: bool = True, verbose: bool = True) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        gs = self.env.grid_size
        self.payoff_table = np.full((gs, gs), np.inf)
        start_time = timeit.default_timer() if timed else None
        pbar = tqdm(desc="Training", total=gs**2, leave=False)
        for ex, ey in np.ndindex(gs, gs):
            env = self.env.__class__(grid_size=gs)
            env.reset(options={"evader_pos": (ex, ey)})
            payoff = self._optimal_playout(env)
            pbar.update()
            if payoff <= self.max_payoff:
                self.payoff_table[ex, ey] = payoff
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

    def _optimal_playout(self, env: HamstrungSquadEnv) -> int:
        """Computes the optimal number of steps to capture the evader."""
        obs = env._get_obs()
        queue = deque([(obs, 0)])
        visited = set([obs])
        payoffs = []
        while queue:
            (obs, payoff) = queue.popleft()
            if np.linalg.norm(np.array(obs[0:2]) - np.array(obs[2:4])) <= 1.5:
                payoffs.append(payoff)
            if payoff >= self.max_payoff:
                continue
            for pursuer_act in range(2):
                for evader_act in range(4):
                    action = (pursuer_act, evader_act)
                    new_obs, _, terminated, truncated, _ = env.step(action, obs)
                    if terminated:
                        payoffs.append(payoff + 1)
                    elif not truncated and new_obs not in visited:
                        queue.append((new_obs, payoff + 1))
                        visited.add(new_obs)
        return min(payoffs)

    def _show_payoff_table(self):
        print("Payoff table:")
        for row in self.payoff_table:
            print(" ".join(f"{x:3.0f}" if x != np.inf else "  -" for x in row))
