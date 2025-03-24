"""
run.py
Run the tasks
"""

from utils import (
    AttractiveField,
    Goal,
    MovingPointObstacle,
    PointRobot,
    RepulsiveRadialField,
    RepulsiveVirtualPeriphery,
    Scene,
    TriangularObstacle,
    Tunnel,
    TunnelField,
)


def main():
    scene = Scene(display_size=(1400, 400), elasticity=0.5, time_step=0.2, sub_steps=10)
    gy = scene.ground_y

    robot = PointRobot(position=(10, gy - 300), mass=1, vmax=100)

    # Task-1
    goal = Goal(position=(1000, gy - 3))
    field_0 = AttractiveField(k_p=0.001, body=goal)
    goal.field = field_0

    # Task-2
    obstacle_1 = TriangularObstacle(base=70, height=60, position=(200, gy - 10))
    field_1 = RepulsiveVirtualPeriphery(k_r=2e3, d0=50, body=obstacle_1)
    obstacle_1.field = field_1

    # Task-3
    obstacle_2 = MovingPointObstacle(position=(600, gy - 3), velocity=(-5, 0))
    field_2 = RepulsiveRadialField(k_r=10, d0=50, body=robot)

    def toggle_field_2():
        start = obstacle_1.position.x + field_1.d0 + field_2.d0 < robot.position.x
        end = obstacle_2.position.x - field_2.d0 < robot.position.x
        if start and not end:
            if robot.field is None:
                robot.field = field_2
                scene.attach_effects(robot, [field_2])
                scene.attach_effects(obstacle_2, [field_2])
        elif end:
            if robot.field is not None:
                robot.field = None
                scene.detach_effects(robot, [field_2])
                scene.detach_effects(obstacle_2, [field_2])
                robot._set_position((obstacle_2.position.x, robot.position.y))

    # Task-4
    tunnel = Tunnel(position=(800, gy - 150), dimensions=(250, 100))
    field_3 = TunnelField(strength=100, body=tunnel)
    tunnel.field = field_3

    # Task-5
    def toggle_goal_velocity():
        start = tunnel.position.x + tunnel.dimensions.x / 2 < robot.position.x
        end = goal.position.x <= robot.position.x
        if start and not end:
            if goal.velocity.length == 0:
                goal._set_velocity((25, 0))
        elif end:
            if goal.velocity.length > 0:
                goal._set_velocity((0, 0))
                robot._set_velocity((0, 0))

    scene.add_bodies([goal, obstacle_1, obstacle_2, tunnel, robot])
    scene.attach_effects(robot, [field_0, field_1, field_3])
    scene.add_pipelines([toggle_field_2, toggle_goal_velocity])

    stopping_condition = lambda: goal.position.x - 7 <= robot.position.x
    scene.render(stopping=stopping_condition, framerate=60)


if __name__ == "__main__":
    main()
