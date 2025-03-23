"""
run.py
Run the tasks
"""

from utils import (
    AttractiveField,
    Goal,
    MovingObstacle,
    PointRobot,
    RepulsiveField,
    Scene,
    StaticObstacle,
    Tunnel,
    TunnelField,
)


def main():
    scene = Scene(display_size="full", elasticity=0.6, dt=0.2, steps=10)
    gy = scene.ground_y

    robot = PointRobot(position=(10, gy - 300), mass=1, vmax=100)

    goal = Goal(position=(1000, gy - 3))
    field_0 = AttractiveField(k_p=0.0005, body=goal)
    goal.field = field_0

    obstacle_1 = StaticObstacle(position=(200, gy - 10), radius=8)
    field_1 = RepulsiveField(k_r=10, d0=50, body=obstacle_1)
    obstacle_1.field = field_1

    obstacle_2 = MovingObstacle(position=(600, gy - 3), velocity=(-5, 0))
    field_2 = RepulsiveField(k_r=10, d0=50, body=robot)

    def toggle_field_2():
        crossed_1 = obstacle_1.position.x + field_1.d0 + field_2.d0 < robot.position.x
        crossed_2 = obstacle_2.position.x - field_2.d0 < robot.position.x
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
                robot._set_position((obstacle_2.position.x, robot.position.y))

    tunnel = Tunnel(position=(800, gy - 150), dimensions=(250, 100))
    field_3 = TunnelField(strength=100, body=tunnel)
    tunnel.field = field_3

    scene.add_bodies([robot, goal, obstacle_1, obstacle_2, tunnel])
    scene.attach_effects(robot, [field_0, field_1, field_3])
    scene.add_pipelines([toggle_field_2])

    scene.render()


if __name__ == "__main__":
    main()
