"""
run.py
Run the Q-Learning agent on the GridWorld environment
"""

import gymnasium as gym
from reinforcement.q_learning import QLearningAgent


def main():
    env = gym.make(
        "GridWorld-v0",
        rewards={"goal": 1000, "obstacle": -200, "default": -1},
        slippage=None,
        obstacle_penalty=None,
        render_mode="ansi",
        seed=25,
    )
    agent = QLearningAgent(
        env,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.9,
        initial_q_table=None,
    )
    agent.train(
        episodes=None,
        threshold=1e-4,
        decay_epsilon=lambda eps: max(0.1, eps * 0.99),
    )
    agent.test()
    env.render()


if __name__ == "__main__":
    main()
