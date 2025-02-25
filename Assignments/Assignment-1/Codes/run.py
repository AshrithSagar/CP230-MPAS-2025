"""
run.py
Run the Q-Learning agent on the GridWorld environment
"""

from grid_world import GridWorld
from q_learning import QLearning


def main():
    env = GridWorld(
        size=(40, 40),
        start=(0, 0),
        goal=[(38, 38), (38, 39), (39, 38), (39, 39)],
        obstacles=[
            [(8, 8), (8, 9), (9, 8), (9, 9), (10, 8)],
            [(18, 18), (18, 19), (19, 18), (19, 19), (20, 18), (19, 17)],
            [(25, 25), (25, 26), (26, 25), (26, 26), (27, 25)],
            [(30, 10), (30, 11), (31, 10), (31, 11), (32, 10)],
            [(35, 35), (35, 36), (36, 35), (36, 36), (37, 35)],
        ],
        rewards={"goal": 100, "obstacle": -100, "default": -1},
    )
    print(env.render())
    agent = QLearning(env, alpha=0.1, gamma=0.9, epsilon=0.1, initial_q_table=None)
    agent.train(episodes=1000)


if __name__ == "__main__":
    main()
