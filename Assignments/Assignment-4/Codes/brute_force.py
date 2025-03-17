"""
brute_force.py
Brute-force agent for the Hamstrung squad game
"""

import timeit
from typing import Any, Dict

import numpy as np
from hamstrung_squad import ActType, Coord, HamstrungSquadEnv
from tqdm import tqdm


class BruteForceAgent:
    """Brute-force agent for the Hamstrung squad game"""

    def __init__(self, env: HamstrungSquadEnv):
        self.env = env

    def _evaluate_action(self, action: ActType) -> float:
        """Evaluate the given action by simulating the environment step"""
        original_state = self.env.__dict__.copy()
        _, reward, _, _, _ = self.env.step(action)
        self.env.__dict__ = original_state
        return reward

    def _select_best_action(self) -> ActType:
        """Select the best action by evaluating all possible actions"""
        best_action = None
        best_reward = -np.inf
        for pursuer_action, evader_action in np.ndindex(2, 4):
            action = (pursuer_action, evader_action)
            reward = self._evaluate_action(action)
            if reward > best_reward:
                best_reward = reward
                best_action = action
        return best_action

    def _train_evader(self, evader: Coord, render_mode: str = None) -> None:
        """Train the agent for a single episode"""
        obs, _ = self.env.reset(evader)
        if render_mode:
            self.env.render()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action: ActType = self._select_best_action()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            if render_mode:
                if render_mode == "ansi":
                    print(f"{obs} -> {action} -> {reward}")
                self.env.render()
            obs = next_obs
        if render_mode == "ansi":
            print(f"{obs} -> {'Terminated' if terminated else 'Truncated'}")

    def train(
        self,
        render: bool = False,
        timed: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train the agent"""
        info: Dict[str, Any] = {}
        start_time = timeit.default_timer() if timed else None
        show_pbar = render and self.env.render_mode != "ansi"
        if show_pbar:
            pbar = tqdm(desc="Training", total=self.env.grid_size**2, leave=False)
        for evader in np.ndindex(self.env.grid_size, self.env.grid_size):
            self._train_evader(
                evader=evader, render_mode=self.env.render_mode if render else None
            )
            if show_pbar:
                pbar.update()
        if show_pbar:
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
