"""
run.py \n
Run the co-ordination algorithm.
"""

from utils import GridMap, Robot, Scene, set_seed


def main():
    set_seed(24233)

    # 40x40 grid, and 6 obstacles each occupying 100 cells
    grid_map = GridMap(grid_size=40, num_obstacles=6, obstacle_occupancy=100)
    robots = Robot.from_count(count=2, start=(0, 0), sensor_range=6)

    scene = Scene(grid_map, robots)
    scene.setup()
    scene.render(
        num_iterations=100, delay_interval=1e-2, close_after=False, record=False
    )
    # Note: record=True needs ffmpeg installed


if __name__ == "__main__":
    main()
