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

    robot = PointRobot(position=(10, gy - 300), mass=1, vmax=40)

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
    field_2 = RepulsiveRadialField(k_r=1e7, d0=100, body=robot)

    def toggle_field_2():
        start = obstacle_1.position.x + field_1.d0 + field_2.d0 < robot.position.x
        end = (obstacle_2.position.x - 7 < robot.position.x) or (robot.position.x > 600)
        if start and not end:
            if robot.field is None:
                robot.field = field_2
                scene.attach_effects(obstacle_2, [field_2])
        elif end:
            if robot.field is not None:
                robot.field = None
                scene.detach_effects(obstacle_2, [field_2])
                obstacle_2._set_velocity((0, 0))

    # Task-4
    tunnel = Tunnel(position=(850, gy - 150), dimensions=(250, 100))
    field_3 = TunnelField(strength=100, body=tunnel)

    def navigate_tunnel():
        tunnel_start, _, tunnel_end, tunnel_top = tunnel._get_dimensions()
        if tunnel_start - 100 <= robot.position.x <= tunnel_end:
            if robot.position.y > tunnel.position.y:
                robot.apply_force_at_local_point((0, -1e3))
            if robot.position.x < tunnel.position.x:
                robot._set_velocity(
                    (min(robot.velocity.x + 1, robot.vmax), robot.velocity.y)
                )
            elif robot.position.x > tunnel.position.x:
                robot._set_velocity((max(robot.velocity.x - 1, 0), robot.velocity.y))
            if robot.position.y < tunnel_top - 10:
                robot._set_velocity((robot.velocity.x, max(robot.velocity.y, 0)))

    # Task-5
    def toggle_goal_velocity():
        _, _, tunnel_end, _ = tunnel._get_dimensions()
        start = tunnel_end - 20 < robot.position.x
        end = goal.position.x <= robot.position.x
        if start and not end:
            if goal.velocity.length == 0:
                goal._set_velocity((20, 0))
        elif end:
            if goal.velocity.length > 0:
                goal._set_velocity((0, 0))
                robot._set_velocity((0, 0))

    scene.add_bodies([goal, obstacle_1, obstacle_2, tunnel, robot])
    scene.attach_effects(robot, [field_0, field_1])
    scene.add_fields([field_3])
    scene.add_pipelines([toggle_field_2, navigate_tunnel, toggle_goal_velocity])

    stopping_condition = lambda: goal.position.x - 7 <= robot.position.x
    scene.render(stopping=stopping_condition, framerate=30, record=True)


if __name__ == "__main__":
    main()
