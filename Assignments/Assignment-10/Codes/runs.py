"""
runs.py \n
Run the co-ordination algorithm accross multiple seeds
"""

import os

from utils import GridMap, Robot, Scene, set_seed


def run(sim_dir: str, seed: int):
    set_seed(seed)

    # 40x40 grid, and 6 obstacles each occupying 100 cells
    grid_map = GridMap(grid_size=40, num_obstacles=6, obstacle_occupancy=100)
    robots = Robot.from_count(count=2, start=(0, 0), sensor_range=6)

    scene = Scene(grid_map, robots)
    scene.setup()
    scene.render(
        num_iterations=100,
        delay_interval=1e-2,
        close_after=True,
        record=True,
        save_file=f"{sim_dir}/simulation-{seed}.mp4",
    )
    # Note: record=True needs ffmpeg installed


def main():
    sim_dir = "simulations"
    os.makedirs(sim_dir, exist_ok=True)

    for seed in [0, 25, 42, 100]:
        run(sim_dir, seed)


if __name__ == "__main__":
    main()
