"""
run.py \n
Run the co-ordination algorithm.
"""

from utils import GridMap, Robot, Scene, set_seed


def main():
    set_seed(24233)

    # 40×40, 6 obstacles of ~10×10 => 100 cells each
    grid_map = GridMap(grid_size=40, num_obstacles=6, obstacle_size=3)
    robots = Robot.from_count(count=2, start=(0, 0), sensor_range=6)

    scene = Scene(grid_map, robots)
    scene.setup()
    scene.render(num_iterations=100, delay_interval=1e-2, close_after=False)


if __name__ == "__main__":
    main()
