"""
run.py
Run the Hamstrung squad game with the agents
"""

from brute_force import BruteForceAgent
from hamstrung_squad import HamstrungSquadEnv
from q_learning import QLearningAgent


def _q_learning(env: HamstrungSquadEnv) -> None:
    agent = QLearningAgent(
        env,
        alpha=0.5,
        gamma=0.99,
        epsilon=0.9,
        initial_q_table=None,
    )
    agent.train(
        episodes=None,
        threshold=1e-4,
        decay_epsilon=lambda eps: max(0.1, eps * 0.99),
        render=True,
        timed=True,
        verbose=True,
    )


def main():
    env = HamstrungSquadEnv(
        grid_size=7,
        max_payoff=10,
        rewards={"capture": 10, "default": -1},
        render_mode="ansi",
        seed=42,
    )
    agent = BruteForceAgent(env, max_payoff=10)
    agent.train(timed=True, verbose=True)


if __name__ == "__main__":
    main()
