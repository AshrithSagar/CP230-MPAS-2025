"""
run.py
Run the tasks
"""

from utils import (
    AttractiveField,
    MovingObstacle,
    PointRobot,
    RepulsiveField,
    Scene,
    StaticObstacle,
)


def main():
    scene = Scene(display_size="full", elasticity=0.6, dt=0.2, steps=10)
    gy = scene.ground_y

    robot = PointRobot(position=(10, gy - 300), velocity=(0, 0), mass=1, vmax=10)
    field_0 = AttractiveField(goal=(1000, gy - 3), k_p=0.0005)

    field_1 = RepulsiveField(k_r=10, d0=50)
    obstacle_1 = StaticObstacle(position=(200, gy - 10), field=field_1, radius=8)
    field_1.obstacle = obstacle_1

    obstacle_2 = MovingObstacle(position=(800, gy - 3), velocity=(-2, 0))

    scene.add_bodies([robot, obstacle_1, obstacle_2])
    scene.add_fields([field_0])
    scene.attach_effects(robot, [field_0, field_1])
    scene.render()


if __name__ == "__main__":
    main()
