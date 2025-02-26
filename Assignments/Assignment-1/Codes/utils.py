"""
utils.py
Utility functions
"""

import matplotlib.pyplot as plt
import numpy as np


def save_q_table(agent, filename="q_table.txt"):
    np.savetxt(filename, np.round(agent.q_table, 2), delimiter=",", fmt="%.2f")


def show_q_table_heatmap(env, agent):
    plt.imshow(agent.q_table.max(axis=1).reshape(env.size), cmap="hot")
    plt.colorbar()
    plt.title("Maximum Q-value for each state")
    plt.show()


def show_q_table_actions(env, agent):
    actions = ["→", "↓", "←", "↑"]
    optimal_actions = np.argmax(agent.q_table, axis=1).reshape(env.size)
    action_grid = np.vectorize(lambda x: actions[x])(optimal_actions)
    for row in action_grid:
        print(" ".join(row))
