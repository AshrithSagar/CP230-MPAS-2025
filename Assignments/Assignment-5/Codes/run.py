"""
run.py
Run the tasks
"""

from utils import AttractiveField, PointRobot


def main():
    robot = PointRobot(mass=1.0, position=(0, 0), velocity=0.0, vmax=10.0)
    field = AttractiveField(dimensions=(100, 100), goal=(20, 0), epsilon=0.1)


if __name__ == "__main__":
    main()
