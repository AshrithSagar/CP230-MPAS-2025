"""
plot.py \n
Plot the velocity obstacle calculations
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

Vec2 = Tuple[float, float]


class Robot:
    def __init__(
        self, position: Vec2, radius: float, speed: float, circling_around: Vec2 = None
    ):
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.speed = speed
        self.circling = circling_around is not None
        self.circling_around = (
            np.array(circling_around, dtype=float) if self.circling else None
        )
        # For circular motion, assume omega = speed / radius.
        self.omega = speed / radius if self.circling else 0

    def circular_position(self, theta):
        return self.circling_around + self.radius * np.array(
            [np.cos(theta), np.sin(theta)]
        )

    def velocity_at_theta(self, theta):
        # The derivative of position yields a tangent vector.
        return self.omega * self.radius * np.array([-np.sin(theta), np.cos(theta)])


class VelocityObstacle:
    """
    Class to compute the velocity obstacle (VO) for two robots.\n
    The VO is a cone that represents the possible velocities for robot A to avoid colliding with robot B.
    """

    def __init__(self, robotA: Robot, robotB: Robot):
        self.safety_radius = robotA.radius + robotB.radius

    def half_angle(self, A, B):
        """Compute the half-angle of the VO cone given positions A and B."""
        D = B - A
        d = np.linalg.norm(D)
        return (
            np.arcsin(self.safety_radius / d) if d > self.safety_radius else np.pi / 2
        )

    def plot_vo(self, ax, A, B, half_angle, color="gray"):
        """Plot the two boundary rays of the VO cone at point A with target B."""
        D = B - A
        d = np.linalg.norm(D)
        angle_center = np.arctan2(D[1], D[0])
        angles = [angle_center - half_angle, angle_center + half_angle]
        for a in angles:
            # Extend rays for visualization (1.5*d)
            ray = A.reshape(2, 1) + np.array([[np.cos(a)], [np.sin(a)]]) * d * 1.5
            ax.plot([A[0], ray[0, 0]], [A[1], ray[1, 0]], color=color, linestyle="--")


class DynamicPlot:
    """Class to animate the simulation over time."""

    def __init__(
        self,
        bodies: Tuple[Robot, Robot],
        avoid_times: List[float],
        total_time: float = 20,
        time_step: float = 0.1,
    ):
        self.robotA, self.robotB = bodies
        self.avoid_times = avoid_times
        self.total_time = total_time
        self.dt = time_step
        self.times = np.arange(0, total_time, time_step)
        self.vo = VelocityObstacle(self.robotA, self.robotB)
        self.A_positions = [self.robotA.position.copy()]
        self.current_time = 0

    def compute_avoidance_velocity(self, A_pos, B_pos, vB):
        """
        At each time step, compute robot A's desired velocity.
        If the desired velocity (toward B) lies inside the VO cone, then
        choose the boundary velocity. Otherwise, use the desired velocity.
        """
        D = B_pos - A_pos
        if np.linalg.norm(D) == 0:
            return np.zeros(2)
        angle_D = np.arctan2(D[1], D[0])
        vA_des = self.robotA.speed * D / np.linalg.norm(D)
        half_angle = self.vo.half_angle(A_pos, B_pos)
        # Compute relative desired velocity
        v_rel = vA_des - vB
        rel_angle = np.arctan2(v_rel[1], v_rel[0])
        # Check if the difference between line-of-sight and relative velocity is within half_angle.
        if abs(rel_angle - angle_D) < half_angle:
            rel_angle_boundary = angle_D + half_angle  # Choose the boundary
            v_rel_boundary = self.robotA.speed * np.array(
                [np.cos(rel_angle_boundary), np.sin(rel_angle_boundary)]
            )
            vA = v_rel_boundary + vB
        else:
            vA = vA_des
        return vA

    def update(
        self, frame, scat_A: plt.scatter, scat_B: plt.scatter, quiv_A: plt.quiver
    ):
        self.current_time += self.dt

        # Update Robot B position on circle.
        theta = np.pi / 2 - self.robotB.omega * self.current_time
        B_pos = self.robotB.circular_position(theta)
        vB = self.robotB.velocity_at_theta(theta)

        # Compute Robot A's avoidance velocity.
        A_pos = self.A_positions[-1]
        vA = self.compute_avoidance_velocity(A_pos, B_pos, vB)

        # Update Robot A's position.
        new_A_pos = A_pos + vA * self.dt
        self.A_positions.append(new_A_pos)

        # Update scatter plots.
        scat_A.set_offsets(new_A_pos.reshape(1, -1))
        scat_B.set_offsets(B_pos.reshape(1, -1))
        # Update quiver for Robot A's velocity vector.
        quiv_A.set_offsets(new_A_pos.reshape(1, -1))
        quiv_A.set_UVC(vA[0], vA[1])

        return scat_A, scat_B, quiv_A

    def run(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Simulation")
        ax.axis("equal")
        ax.grid(True)

        # Plot Robot B's circular path.
        theta_vals = np.linspace(0, 2 * np.pi, 200)
        circle_B = self.robotB.circling_around.reshape(
            2, 1
        ) + self.robotB.radius * np.array([np.cos(theta_vals), np.sin(theta_vals)])
        ax.plot(circle_B[0, :], circle_B[1, :], "r--", alpha=0.5, label="B's path")

        # Initial positions.
        scat_A = ax.scatter(
            self.A_positions[0][0],
            self.A_positions[0][1],
            color="blue",
            s=50,
            label="Robot A",
        )
        theta0 = np.pi / 2  # initial theta for robot B
        B0 = self.robotB.circular_position(theta0)
        scat_B = ax.scatter(B0[0], B0[1], color="red", s=50, label="Robot B")
        quiv_A = ax.quiver(
            self.A_positions[0][0],
            self.A_positions[0][1],
            0,
            0,
            color="blue",
            scale=1,
            scale_units="xy",
        )
        ax.legend()

        anim = FuncAnimation(
            fig,
            self.update,
            fargs=(scat_A, scat_B, quiv_A),
            frames=len(self.times),
            interval=self.dt * 1000,
            blit=False,
        )
        plt.show()


def main():
    robotA = Robot(position=(0, 0), radius=1, speed=2)
    robotB = Robot(position=(10, 2), radius=3, speed=1.5, circling_around=(10, 0))

    scene = DynamicPlot(
        bodies=[robotA, robotB], avoid_times=[2, 6, 10], total_time=20, time_step=0.1
    )
    scene.run()


if __name__ == "__main__":
    main()
