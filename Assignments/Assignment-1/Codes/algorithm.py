"""
algorithm.py
Algorithm implementations.
"""

import random

import numpy as np
from grid import create_grid


class QLearning:
    """
    Class to implement Q-Learning algorithm.
    """

    def __init__(
        self, grid, start, goal_area, obstacles, alpha=0.1, gamma=0.6, epsilon=0.1
    ):
        self.grid = grid
        self.start = start
        self.goal_area = goal_area
        self.obstacles = obstacles

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros((len(grid), len(grid), 4))

    def get_next_action(self, state):
        """
        Get the next action based on epsilon-greedy policy
        """
        if random.uniform(0, 1) < self.epsilon:
            # Random action | Exploration
            return random.randint(0, 3)
        else:
            # Greedy action | Exploitation
            return np.argmax(self.q_table[state])

    def get_neighbours(self, state):
        """
        Get the neighbours of the current state
        """
        x, y = state
        neighbours = []
        if x > 0:
            neighbours.append((x - 1, y))  # Left
        if x < len(self.grid) - 1:
            neighbours.append((x + 1, y))  # Right
        if y > 0:
            neighbours.append((x, y - 1))  # Up
        if y < len(self.grid) - 1:
            neighbours.append((x, y + 1))  # Down
        return neighbours

    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-Table based on Bellman Equation
        """
        self.q_table[state][action] += self.alpha * (
            reward
            + self.gamma * np.max(self.q_table[next_state])
            - self.q_table[state][action]
        )

    def train(self, episodes=1000, verbose=False):
        for episode in range(episodes):
            state = self.start
            total_reward = 0
            steps = 0
            while state not in self.goal_area:
                action = self.get_next_action(state)
                next_state = self.move(state, action)
                reward = self.get_reward(next_state)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                steps += 1
                if verbose:
                    print(f"State: {state}, Action: {action}, Reward: {reward}")
            if verbose:
                print(
                    f"Episode {episode + 1}: Total Reward: {total_reward}, Steps: {steps}"
                )

    def move(self, state, action):
        x, y = state
        if action == 0:
            x -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x += 1
        elif action == 3:
            y -= 1

        # Ensure within grid
        x = max(0, min(x, len(self.grid) - 1))
        y = max(0, min(y, len(self.grid) - 1))

        return x, y

    def get_reward(self, state):
        if state in self.goal_area:
            return 100
        elif state in self.obstacles:
            return -100
        else:
            return -1

    def get_path(self):
        path = []
        state = self.start
        while state not in self.goal_area:
            action = np.argmax(self.q_table[state])
            next_state = self.move(state, action)
            path.append(next_state)
            state = next_state

        return path


if __name__ == "__main__":
    grid, start, goal_area, obstacles = create_grid()

    q_learning = QLearning(grid, start, goal_area, obstacles)
    q_learning.train(verbose=True)
    path = q_learning.get_path()
    print(path)
    print(q_learning.q_table)
