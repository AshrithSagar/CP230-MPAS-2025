"""
run.py
Run the tasks
"""

from utils import AttractiveField, PointRobot, Scene


def main():
    robot = PointRobot(mass=1.0, position=(0, 5), velocity=(0, 0), vmax=1.0)
    field = AttractiveField(goal=(20, 0), k_p=0.1)
    scene = Scene(time_step=0.1, epsilon=1.0)
    scene.add_body(robot)
    scene.render()


if __name__ == "__main__":
    main()
