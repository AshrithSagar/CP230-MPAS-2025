"""
brute_force.py
Brute-force agent for the Hamstrung squad game
"""

import timeit
from typing import Any, Dict

import numpy as np
from hamstrung_squad import Coord, HamstrungSquadEnv, ObsType
from tqdm import tqdm


class BruteForceAgent:
    """Brute-force agent with minimax search"""

    def __init__(self, env: HamstrungSquadEnv, max_payoff: int = 10) -> None:
        self.env = env
        self.max_depth = max_payoff
        self.memo = {}

    def _minimax(self, obs: ObsType, depth: int, is_evader: bool) -> int:
        """Minimax search with memoization."""
        if tuple(obs) in self.memo:
            return self.memo[tuple(obs)]
        if np.linalg.norm(np.array(obs[:2]) - np.array(obs[2:4])) <= 1.5:
            return 0  # Immediate capture
        best_payoff = -self.max_depth if is_evader else self.max_depth
        if depth == 0:
            return best_payoff
        for action in np.ndindex(2, 4):
            next_obs, _, _, _, _ = self.env.step(action, simulate=True)
            payoff = self._minimax(tuple(next_obs), depth - 1, not is_evader)
            _operation = max if is_evader else min
            best_payoff = _operation(best_payoff, payoff)
        self.memo[tuple(obs)] = best_payoff
        return best_payoff

    def _train_evader(self, evader: Coord) -> None:
        """Train the agent for a given evader position."""
        obs, _ = self.env.reset(evader)
        payoff = self._minimax(obs, self.max_depth, is_evader=False)
        self.env.payoff_table[tuple(evader)] = payoff

    def train(self, timed: bool = True, verbose: bool = True) -> Dict[str, Any]:
        """Train the agent"""
        info: Dict[str, Any] = {}
        start_time = timeit.default_timer() if timed else None
        pbar = tqdm(desc="Training", total=self.env.grid_size**2, leave=False)
        for evader in np.ndindex(self.env.grid_size, self.env.grid_size):
            self._train_evader(evader=evader)
            pbar.update()
        pbar.close()
        if timed:
            elapsed = timeit.default_timer() - start_time
            info["time"] = elapsed
        if verbose:
            print("Training completed:")
            if timed:
                print(f" Time: {elapsed:.3f}s")
            self.env._show_payoff_table()
        return info
