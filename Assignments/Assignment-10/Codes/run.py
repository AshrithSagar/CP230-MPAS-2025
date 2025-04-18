"""
run.py \n
Run the co-ordination algorithm.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from utils import Coordinator, GridMap, Point, Robot


def main():
    # --- Setup ---
    # Random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # 40×40, 6 obstacles of ~10×10 => 100 cells each
    grid_map = GridMap(size=40, num_obstacles=6, obs_size=10)
    start: Point = (0, 0)
    robots = [Robot(1, start, 6), Robot(2, start, 6)]
    coord = Coordinator(grid_map, robots)

    # Initial sensing
    for r in robots:
        grid_map.mark_explored(r.pos, r.sensor_range)

    # Prepare figure once
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = ListedColormap(["lightgray", "white", "yellow", "dimgray"])
    im = ax.imshow(np.zeros((grid_map.N, grid_map.N)), origin="lower", cmap=cmap)
    robot_colors = plt.cm.tab10(np.linspace(0, 1, len(robots)))
    texts = []  # Will hold utility‐labels
    lines = []  # Will hold path lines
    dots = []  # Will hold robot markers

    # --- Animation loop ---
    for t in range(1, 11):
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
            if len(r.path) > 1:
                xs, ys = zip(*r.path)
                (ln,) = ax.plot(ys, xs, "--", linewidth=1, color=color)
                lines.append(ln)
            px, py = r.pos
            (dot,) = ax.plot(py, px, "o", label=f"R{r.id}", color=color)
            dots.append(dot)

        ax.set_title(f"Iteration {t}")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw & pause (non‐blocking)
        fig.canvas.draw_idle()
        plt.pause(0.5)

        # Now move robots and re‐sense
        # Each robot moves one step and re‑explores
        for r in robots:
            r.step()
            grid_map.mark_explored(r.pos, r.sensor_range)
        print(f"Step {t}: positions = {[r.pos for r in robots]}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
