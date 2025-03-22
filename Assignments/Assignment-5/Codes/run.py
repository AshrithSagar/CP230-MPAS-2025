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

    obstacle_1 = StaticObstacle(position=(200, gy - 10), radius=8)
    field_1 = RepulsiveField(k_r=10, d0=50, body=obstacle_1)
    obstacle_1.field = field_1

    obstacle_2 = MovingObstacle(position=(800, gy - 3), velocity=(-2, 0))

    scene.add_bodies([robot, obstacle_1, obstacle_2])
    scene.add_fields([field_0])
    scene.attach_effects(robot, [field_0, field_1])

    field_2 = RepulsiveField(k_r=10, d0=50, body=robot)

    def toggle_field_2():
        crossed_1 = obstacle_1.position.x + field_1.d0 + field_2.d0 < robot.position.x
        crossed_2 = obstacle_2.position.x + field_2.d0 < robot.position.x
        if crossed_1 and not crossed_2:
            if robot.field is None:
                robot.field = field_2
                scene.attach_effects(robot, [field_2])
                scene.attach_effects(obstacle_2, [field_2])
        else:
            if robot.field is not None:
                robot.field = None
                scene.detach_effects(robot, [field_2])
                scene.detach_effects(obstacle_2, [field_2])

    scene.add_pipeline(toggle_field_2)

    scene.render()


if __name__ == "__main__":
    main()
