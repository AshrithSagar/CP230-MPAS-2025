"""
run.py \n
Run the co-ordination algorithm.
"""

from utils import Coordinator, GridMap, Point, Robot


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

        # Each robot moves one step and re‑explores
        for r in R:
            r.step()
            G.mark_explored(r.pos, r.sensor_range)
        print(f"Step {t+1}: positions = {[r.pos for r in R]}")


if __name__ == "__main__":
    main()
