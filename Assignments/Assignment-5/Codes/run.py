"""
run.py
Run the tasks
"""

from utils import AttractiveField, Obstacle, PointRobot, RepulsiveField, Scene


def main():
    scene = Scene(display_size="full", elasticity=0.8, dt=0.2, steps=10)
    gy = scene.ground_y

    robot = PointRobot(position=(100, gy - 450), velocity=(0, 0), mass=1, vmax=10)
    obstacle_1 = Obstacle(position=(400, gy - 10), radius=10, d0=25)
    scene.add_bodies([robot, obstacle_1])

    field_0 = AttractiveField(goal=(600, gy), k_p=0.001)
    field_1 = RepulsiveField(obstacle_1, k_r=0.001)
    scene.attach_fields(robot, [field_0, field_1])

    scene.render()


if __name__ == "__main__":
    main()
