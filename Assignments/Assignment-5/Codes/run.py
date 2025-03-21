"""
run.py
Run the tasks
"""

from utils import AttractiveField, Obstacle, PointRobot, Scene


def main():
    scene = Scene(display_size="full", elasticity=0.8, dt=0.2, steps=10)
    gy = scene.ground_y

    robot = PointRobot(position=(100, gy - 450), velocity=(0, 0), mass=1, vmax=10)
    obstacle = Obstacle(position=(400, gy - 10), radius=10)

    field = AttractiveField(goal=(1000, gy), k_p=0.8)
    scene.apply_field(field, robot)

    scene.add_bodies([robot, obstacle])
    scene.render()


if __name__ == "__main__":
    main()
