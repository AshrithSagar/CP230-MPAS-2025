"""
run.py \n
Run the co-ordination algorithm.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from utils import Coordinator, GridMap, Point, Robot


def visualize(grid: GridMap, coord: Coordinator, robots: list, step: int):
    # Get state and frontier list
    state, fronts = grid.grid_state()
    arr = np.array(state)

    # Build colormap: 0=lightgray,1=white,2=yellow,3=dimgray
    cmap = ListedColormap(["lightgray", "white", "yellow", "dimgray"])
    plt.figure(figsize=(6, 6))
    plt.imshow(arr, origin="lower", cmap=cmap, interpolation="none")

    # Annotate utilities at each frontier
    for x, y in fronts:
        u = coord.U.get((x, y), 0.0)
        plt.text(y, x, f"{u:.2f}", ha="center", va="center", fontsize=6, color="black")

    # Plot robot paths & positions
    for r in robots:
        if len(r.path) > 1:
            xs, ys = zip(*r.path)
            plt.plot(ys, xs, "--", linewidth=1)
        # Current position
        px, py = r.pos
        plt.plot(py, px, "o", label=f"R{r.id}")

    plt.title(f"Iteration {step}")
    plt.legend(loc="upper right", fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def main():
    # 40×40, 6 obstacles of ~10×10 => 100 cells each
    G = GridMap(size=40, num_obstacles=6, obs_size=10)
    start: Point = (0, 0)
    R = [Robot(1, start, 6), Robot(2, start, 6)]
    coord = Coordinator(G, R)

    # Initial sensing
    for r in R:
        G.mark_explored(r.pos, r.sensor_range)

    for t in range(10):
        coord.assign()
        visualize(G, coord, R, t + 1)

        # Each robot moves one step and re‑explores
        for r in R:
            r.step()
            G.mark_explored(r.pos, r.sensor_range)
        print(f"Step {t+1}: positions = {[r.pos for r in R]}")


if __name__ == "__main__":
    main()
