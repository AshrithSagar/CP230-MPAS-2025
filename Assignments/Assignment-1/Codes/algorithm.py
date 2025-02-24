"""
algorithm.py
Implementations
"""

import random
from enum import Enum
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

random.seed(42)


class GridWorld:
    """Grid world environment for the Q-learning agent."""

    def __init__(
        self,
        grid_size: int,
        start: List[int],
        goal: List[List[int]],
        obstacles: List[List[int]],
        rewards: dict = None,
    ):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.agent_position: List[int] = start
        default_rewards = {"goal": 10, "obstacle": -10, "step": -1}
        self.rewards: dict = rewards if rewards else default_rewards

    class Action(Enum):
        """Actions that the agent can take in the environment."""

        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3

    def reset(self) -> List[int]:
        """Reset the agent's position to the starting state."""
        self.agent_position = self.start
        return self.agent_position

    def step(self, action: Action) -> Tuple[List[int], int, bool]:
        """Take an action and return the next state, reward, and whether the episode is done."""
        if action == GridWorld.Action.UP:
            self.agent_position[0] -= 1
        elif action == GridWorld.Action.DOWN:
            self.agent_position[0] += 1
        elif action == GridWorld.Action.LEFT:
            self.agent_position[1] -= 1
        elif action == GridWorld.Action.RIGHT:
            self.agent_position[1] += 1

        # Stay within the grid boundaries
        self.agent_position[0] = max(0, min(self.agent_position[0], self.grid_size - 1))
        self.agent_position[1] = max(0, min(self.agent_position[1], self.grid_size - 1))

        # Check if the episode is done
        if self.agent_position in self.goal:
            reward, done = self.rewards["goal"], True
        elif self.agent_position in self.obstacles:
            reward, done = self.rewards["obstacle"], True
        else:
            reward, done = self.rewards["step"], False
        return self.agent_position, reward, done


class QLearningAgent:
    """
    Q-learning agent that learns to navigate the grid world environment.

    Parameters:
    env (GridWorld): The grid world environment.
    alpha (float): Learning rate.
    gamma (float): Discount factor.
    epsilon (float): Exploration rate.
    """

    def __init__(
        self,
        env: GridWorld,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
    ):
        self.env: GridWorld = env
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.q_table: np.ndarray = np.zeros(
            (env.grid_size, env.grid_size, len(GridWorld.Action))
        )

    def choose_action(
        self, state: List[int], policy: str = "epsilon-greedy"
    ) -> GridWorld.Action:
        """
        Choose an action based on a policy

        Parameters:
        state (List[int]): The current state.
        policy (str): The policy to use for choosing the action.
        Can be one of "epsilon-greedy" or "greedy".
        """
        if policy == "epsilon-greedy":
            if random.random() < self.epsilon:
                return GridWorld.Action(random.choice(list(GridWorld.Action)))
            else:
                return GridWorld.Action(np.argmax(self.q_table[state[0], state[1]]))
        elif policy == "greedy":
            return GridWorld.Action(np.argmax(self.q_table[state[0], state[1]]))
        else:
            raise ValueError("Invalid policy")

    def learn(
        self,
        state: List[int],
        action: GridWorld.Action,
        reward: int,
        next_state: List[int],
        done: bool,
    ) -> None:
        """Update the Q-table using the Q-learning update rule."""
        best_next_q_value = np.max(self.q_table[next_state[0], next_state[1]])
        td_target = reward + self.gamma * best_next_q_value * (not done)
        td_error = td_target - self.q_table[state[0], state[1], action.value]
        self.q_table[state[0], state[1], action.value] += self.alpha * td_error

    def train(self, episodes: int) -> None:
        """Train the agent by running Q-learning over multiple episodes."""
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state, policy="epsilon-greedy")
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state

    def evaluate(self, episodes: int) -> float:
        """Evaluate the agent's performance by running it without exploration (greedy policy)."""
        total_reward: float = 0
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state, policy="greedy")
                state, reward, done = self.env.step(action)
                total_reward += reward
        avg_reward: float = total_reward / episodes
        print(f"Average reward over {episodes} episodes: {avg_reward}")
        return avg_reward


def animate(
    agent: QLearningAgent,
    env: GridWorld,
    show_grid: bool = False,
    markersize: int = 10,
    frames: int = 100,
) -> None:
    """Animate the agent's movement in the grid world."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)

    # Grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(show_grid)

    # Mark start, goal, obstacles, and agent
    start_patch = plt.Rectangle(
        (env.start[0], env.start[1]), 1, 1, color="purple", alpha=0.5
    )
    ax.add_patch(start_patch)
    for goal in env.goal:
        goal_patch = plt.Rectangle((goal[0], goal[1]), 1, 1, color="green", alpha=0.5)
        ax.add_patch(goal_patch)
    for obs in env.obstacles:
        obstacle_patch = plt.Rectangle((obs[0], obs[1]), 1, 1, color="red", alpha=0.5)
        ax.add_patch(obstacle_patch)
    agent_marker = plt.plot([], [], marker="o", markersize=markersize, color="blue")[0]

    def update(frame):
        """Update the agent's position for each frame."""
        state = env.reset()
        for _ in range(frame):
            action = agent.choose_action(state, policy="epsilon-greedy")
            state, _, _ = env.step(action)
            agent_marker.set_data([state[1] + 0.5], [state[0] + 0.5])

    _ = FuncAnimation(fig, update, frames=frames, interval=250)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = GridWorld(
        grid_size=40,
        start=[0, 0],
        goal=[[39, 39]],
        obstacles=[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
    )
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.2)
    agent.train(episodes=1000)
    agent.evaluate(episodes=100)
    animate(agent, env, markersize=5)
