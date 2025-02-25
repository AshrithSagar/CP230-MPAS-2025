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
        self.agent_position: List[int] = start[:]
        self.rewards = rewards or {"goal": 10, "obstacle": -10, "step": -1}

    class Action(int, Enum):
        """Actions that the agent can take in the environment."""

        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3

    def reset(self) -> List[int]:
        """Reset the agent's position to the starting state."""
        self.agent_position = self.start[:]
        return self.agent_position[:]

    def step(self, action: Action) -> Tuple[List[int], int, bool]:
        """Take an action and return the next state, reward, and whether the episode is done."""
        prev_position = self.agent_position[:]

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
            self.agent_position = prev_position
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
        q_values = self.q_table[state[0], state[1]]
        if policy == "epsilon-greedy":
            if random.random() < self.epsilon:
                return GridWorld.Action(random.choice(list(GridWorld.Action)))
            else:
                max_q = np.max(q_values)
                max_actions = [
                    action for action, value in enumerate(q_values) if value == max_q
                ]
                chosen_action = random.choice(max_actions)
                return GridWorld.Action(chosen_action)
        elif policy == "greedy":
            max_q = np.max(q_values)
            max_actions = [
                action for action, value in enumerate(q_values) if value == max_q
            ]
            chosen_action = random.choice(max_actions)
            return GridWorld.Action(chosen_action)
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
    """Animate the agent's movement in the grid world, tracing the path taken."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(show_grid)

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

    (agent_marker,) = ax.plot([], [], marker="o", markersize=markersize, color="blue")
    (path_line,) = ax.plot([], [], color="blue", linewidth=2, alpha=0.5)

    state = env.reset()
    optimal_path = [state[:]]
    done = False
    while not done:
        action = agent.choose_action(state, policy="epsilon-greedy")
        next_state, _, done = env.step(action)
        optimal_path.append(next_state[:])
        state = next_state

    x_path = [s[0] + 0.5 for s in optimal_path]
    y_path = [s[1] + 0.5 for s in optimal_path]

    def update(frame):
        if frame < len(x_path):
            agent_marker.set_data([x_path[frame]], [y_path[frame]])
            path_line.set_data(x_path[: frame + 1], y_path[: frame + 1])
        return agent_marker, path_line

    _ = FuncAnimation(fig, update, frames=frames, interval=250, blit=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = GridWorld(
        grid_size=10,
        start=[0, 0],
        goal=[[9, 9]],
        obstacles=[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        rewards={"goal": 10, "obstacle": -10, "step": -1},
    )
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.5)
    agent.train(episodes=1000)
    agent.evaluate(episodes=100)
    animate(agent, env, markersize=10)
