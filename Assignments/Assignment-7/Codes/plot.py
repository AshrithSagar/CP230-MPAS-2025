"""
plot.py \n
Plot the velocity obstacle calculations
"""

import matplotlib.pyplot as plt
import numpy as np


class Robot:
    def __init__(self, position, radius, speed, is_moving_in_circle=False, center=None):
        self.position = np.array(position)
        self.radius = radius
        self.speed = speed
        self.is_moving_in_circle = is_moving_in_circle
        self.center = np.array(center) if center is not None else None
        self.omega = speed / radius if is_moving_in_circle else 0

    def circular_position(self, theta):
        if not self.is_moving_in_circle or self.center is None:
            raise ValueError("Robot is not moving in a circular path.")
        return self.center + self.radius * np.array([np.cos(theta), np.sin(theta)])

    def velocity_at_theta(self, theta):
        if not self.is_moving_in_circle or self.center is None:
            raise ValueError("Robot is not moving in a circular path.")
        return self.omega * self.radius * np.array([-np.sin(theta), np.cos(theta)])


class VelocityObstacle:
    def __init__(self, safety_radius):
        self.safety_radius = safety_radius

    def half_angle(self, A, B):
        D = B - A
        d = np.linalg.norm(D)
        return (
            np.arcsin(self.safety_radius / d) if d > self.safety_radius else np.pi / 2
        )

    def plot_vo(self, ax, A, B, half_angle, color="gray"):
        D = B - A
        d = np.linalg.norm(D)
        angle_center = np.arctan2(D[1], D[0])
        angles = [angle_center - half_angle, angle_center + half_angle]
        for a in angles:
            ray = A.reshape(2, 1) + np.array([[np.cos(a)], [np.sin(a)]]) * d * 1.5
            ax.plot([A[0], ray[0, 0]], [A[1], ray[1, 0]], color=color, linestyle="--")


def main():
    # Parameters
    radius_A = 1
    radius_B = 2
    safety_radius = radius_A + radius_B
    t0, t1, t2 = 0, 10, 15

    # Robots
    robot_A = Robot(position=[0, 0], radius=radius_A, speed=2.0)
    robot_B = Robot(
        position=[10, 0],
        radius=radius_B,
        speed=1.5,
        is_moving_in_circle=True,
        center=[10, 0],
    )
    vo = VelocityObstacle(safety_radius)

    # Robot B positions
    theta0 = np.pi / 2
    B_t0 = robot_B.circular_position(theta0)
    B_t1 = robot_B.circular_position(theta0 - robot_B.omega * t1)
    B_t2 = robot_B.circular_position(theta0 - robot_B.omega * t2)

    # At t0
    D0 = B_t0 - robot_A.position
    angle_D0 = np.arctan2(D0[1], D0[0])
    half_angle0 = vo.half_angle(robot_A.position, B_t0)
    vA_des0 = robot_A.speed * D0 / np.linalg.norm(D0)
    vB_t0 = np.array([-1.5, 0])
    rel_angle_boundary = angle_D0 + half_angle0
    v_rel_boundary = robot_A.speed * np.array(
        [np.cos(rel_angle_boundary), np.sin(rel_angle_boundary)]
    )
    vA_avoid0 = v_rel_boundary + vB_t0

    # At t1
    A_t1 = robot_A.position + vA_avoid0 * 1
    D1 = B_t1 - A_t1
    half_angle1 = vo.half_angle(A_t1, B_t1)
    vA_des1 = robot_A.speed * (D1 / np.linalg.norm(D1))
    theta1 = theta0 - robot_B.omega * t1
    vB_t1 = robot_B.velocity_at_theta(theta1)
    vA_avoid1 = vA_des1

    # At t2
    A_t2 = A_t1 + vA_avoid1 * 1
    D2 = B_t2 - A_t2
    half_angle2 = vo.half_angle(A_t2, B_t2)
    vA_des2 = robot_A.speed * (D2 / np.linalg.norm(D2))
    theta2 = theta0 - robot_B.omega * t2
    vB_t2 = robot_B.velocity_at_theta(theta2)
    vA_avoid2 = vA_des2

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Robot trajectories
    ax.plot(
        [robot_A.position[0], A_t1[0], A_t2[0]],
        [robot_A.position[1], A_t1[1], A_t2[1]],
        "bo-",
        label="Robot A",
    )
    ax.plot(
        [B_t0[0], B_t1[0], B_t2[0]], [B_t0[1], B_t1[1], B_t2[1]], "ro-", label="Robot B"
    )

    # Robot B's path
    theta_vals = np.linspace(0, 2 * np.pi, 200)
    circle_B = robot_B.center.reshape(2, 1) + robot_B.radius * np.array(
        [np.cos(theta_vals), np.sin(theta_vals)]
    )
    ax.plot(circle_B[0, :], circle_B[1, :], "r--", alpha=0.5, label="B's path")

    # VO cones
    vo.plot_vo(ax, robot_A.position, B_t0, half_angle0)
    vo.plot_vo(ax, A_t1, B_t1, half_angle1)
    vo.plot_vo(ax, A_t2, B_t2, half_angle2)

    # Velocity vectors for A
    ax.quiver(
        robot_A.position[0],
        robot_A.position[1],
        vA_avoid0[0],
        vA_avoid0[1],
        color="blue",
        angles="xy",
        scale_units="xy",
        scale=1,
        label="A's v at t0",
    )
    ax.quiver(
        A_t1[0],
        A_t1[1],
        vA_avoid1[0],
        vA_avoid1[1],
        color="green",
        angles="xy",
        scale_units="xy",
        scale=1,
        label="A's v at t1",
    )
    ax.quiver(
        A_t2[0],
        A_t2[1],
        vA_avoid2[0],
        vA_avoid2[1],
        color="purple",
        angles="xy",
        scale_units="xy",
        scale=1,
        label="A's v at t2",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trajectories of Robots A and B and Velocity Obstacles")
    ax.legend()
    ax.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
