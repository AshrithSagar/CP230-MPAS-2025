"""
run.py
Run the tasks
"""

from utils import AttractiveField, PointRobot, Scene


def main():
    scene = Scene(time_step=0.1, elasticity=1.0, display_size=(800, 600))

    robot = PointRobot(mass=1.0, position=(400, 300), velocity=(0, 0), vmax=10)
    scene.add_body(robot)

    field = AttractiveField(goal=(200, 200), k_p=0)
    scene.apply_field(field, robot)

    scene.render()


if __name__ == "__main__":
    main()
