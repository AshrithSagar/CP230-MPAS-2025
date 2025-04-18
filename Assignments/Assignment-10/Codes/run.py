"""
run.py \n
Run the co-ordination algorithm.
"""

import logging
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from utils import Coordinator, GridMap, Robot

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    # --- Setup ---
    # Random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    logger.debug(f"Random seed: {seed}")

    # 40×40, 6 obstacles of ~10×10 => 100 cells each
    grid_map = GridMap(size=40, num_obstacles=6, obs_size=10)
    robots = Robot.from_count(count=2, start=(0, 0), sensor_range=6)
    coord = Coordinator(grid_map, robots)
    num_iterations = 10

    # Initial sensing
    for robot in robots:
        grid_map.mark_explored(robot.pos, robot.sensor_range)

    # Prepare figure once
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    # 0=unknown, 1=explored, 2=frontier, 3=obstacle
    cmap = ListedColormap(["lightgray", "white", "yellow", "dimgray"])
    im = ax.imshow(np.zeros((grid_map.N, grid_map.N)), origin="upper", cmap=cmap)
    robot_colors = plt.cm.tab10(np.linspace(0, 1, len(robots)))
    texts: List[plt.Text] = []  # Will hold utility texts
    lines: List[plt.Line2D] = []  # Will hold path lines
    dots: List[plt.Line2D] = []  # Will hold robot markers

    # --- Animation loop ---
    for t in range(1, num_iterations + 1):
        coord.assign()

        # Update grid image
        state, fronts = grid_map.grid_state()
        im.set_data(np.array(state))

        # Clear old annotations
        for obj in texts + lines + dots:
            obj.remove()
        texts.clear()
        lines.clear()
        dots.clear()

        # Draw utilities on frontiers
        for x, y in fronts:
            u = coord.U.get((x, y), 0.0)
            txt = ax.text(
                y, x, f"{u:.2f}", ha="center", va="center", fontsize=6, color="black"
            )
            texts.append(txt)

        # Draw each robot’s planned path and current position
        for r, color in zip(robots, robot_colors):
            # Plot the entire path history
            xs, ys = zip(*[r.pos] + r.path)
            ax.plot(ys, xs, "-", linewidth=1, color=color, alpha=0.6)  # Persistent path
            # Plot the current position
            px, py = r.pos
            (dot,) = ax.plot(py, px, "o", label=f"R{r.id}", color=color)
            dots.append(dot)

        ax.set_title(f"Iteration {t}")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw & pause (non‐blocking)
        fig.canvas.draw_idle()
        plt.pause(0.1)

        # Each robot moves one step and re‑explores
        for robot in robots:
            robot.step()
            grid_map.mark_explored(robot.pos, robot.sensor_range)

        logger.info(
            f"Step {t}:\n"
            f"  positions:  " + ", ".join(f"Robot {r.id}: {r.pos}" for r in robots)
        )

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
