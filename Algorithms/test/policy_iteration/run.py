"""
run.py
Run the Policy Iteration agent on the GridWorld environment
"""

import gymnasium as gym
from reinforcement.policy_iteration import PolicyIterationAgent


def main():
    env = gym.make(
        "GridWorld-v0",
        rewards={"goal": 1000, "obstacle": -200, "default": -1},
        slippage=None,
        obstacle_penalty=None,
        render_mode="ansi",
        seed=25,
    )
    agent = PolicyIterationAgent(
        env,
        gamma=0.9,
        initial_v_table=None,
    )
    agent.train(
        episodes=None,
        threshold=1e-4,
    )
    agent.test()
    env.render()


if __name__ == "__main__":
    main()
