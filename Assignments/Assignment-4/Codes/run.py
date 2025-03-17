"""
run.py
Run the Hamstrung sqaud game with the Q-learning agent
"""

from hamstrung_squad import HamstrungSquadEnv
from q_learning import QLearningAgent


def main():
    env = HamstrungSquadEnv(
        grid_size=5,
        max_payoff=10,
        rewards={"capture": 10, "default": -1},
        render_mode="ansi",
        seed=42,
    )
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


if __name__ == "__main__":
    main()
