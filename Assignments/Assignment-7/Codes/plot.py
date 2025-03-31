"""
plot.py \n
Plot the velocity obstacle calculations
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class Robot:
    def __init__(self, position, radius, speed, is_moving_in_circle=False, center=None):
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.speed = speed
        self.is_moving_in_circle = is_moving_in_circle
        self.center = np.array(center, dtype=float) if center is not None else None
        # For circular motion, assume omega = speed / radius.
        self.omega = speed / radius if is_moving_in_circle else 0

    def circular_position(self, theta):
        if not self.is_moving_in_circle or self.center is None:
            raise ValueError("Robot is not moving in a circular path.")
        return self.center + self.radius * np.array([np.cos(theta), np.sin(theta)])

    def velocity_at_theta(self, theta):
        if not self.is_moving_in_circle or self.center is None:
            raise ValueError("Robot is not moving in a circular path.")
        # The derivative of position yields a tangent vector.
        return self.omega * self.radius * np.array([-np.sin(theta), np.cos(theta)])


class VelocityObstacle:
    def __init__(self, safety_radius):
        self.safety_radius = safety_radius

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


class StaticPlot:
    """Class to create a fixed-time plot of the simulation."""

    def __init__(self, t_values=[0, 10, 15]):
        # Parameters
        self.radius_A = 1
        self.radius_B = 2
        self.safety_radius = self.radius_A + self.radius_B
        self.t_values = t_values

        self.robot_A = Robot(position=[0, 0], radius=self.radius_A, speed=2.0)
        # Robot B moves in a circle centered at [10, 0].
        self.robot_B = Robot(
            position=[10, 0],
            radius=self.radius_B,
            speed=1.5,
            is_moving_in_circle=True,
            center=[10, 0],
        )
        self.vo = VelocityObstacle(self.safety_radius)

    def plot(self):
        # Compute positions at different time instances.
        # For robot B, use circular_position based on theta.
        theta0 = np.pi / 2  # starting at top of the circle.
        theta_list = [theta0 - self.robot_B.omega * t for t in self.t_values]
        B_positions = [self.robot_B.circular_position(theta) for theta in theta_list]

        # For robot A, compute positions sequentially.
        # At t0, initial position.
        A_positions = [self.robot_A.position.copy()]
        # At t0, compute VO and avoidance velocity.
        D0 = B_positions[0] - self.robot_A.position
        angle_D0 = np.arctan2(D0[1], D0[0])
        half_angle0 = self.vo.half_angle(self.robot_A.position, B_positions[0])
        # Desired velocity (collision course)
        vA_des0 = self.robot_A.speed * D0 / np.linalg.norm(D0)
        # Assume robot B's velocity at t0 is given (example value)
        vB_t0 = np.array([-1.5, 0])
        # Choose boundary on one side of the VO cone:
        rel_angle_boundary = angle_D0 + half_angle0
        v_rel_boundary = self.robot_A.speed * np.array(
            [np.cos(rel_angle_boundary), np.sin(rel_angle_boundary)]
        )
        vA_avoid0 = v_rel_boundary + vB_t0

        # Compute next positions using a 1-second interval.
        A_t1 = A_positions[0] + vA_avoid0 * 1
        A_positions.append(A_t1)
        # At t1, compute new VO parameters.
        D1 = B_positions[1] - A_t1
        half_angle1 = self.vo.half_angle(A_t1, B_positions[1])
        vA_des1 = self.robot_A.speed * (D1 / np.linalg.norm(D1))
        # For robot B, get its tangent velocity at theta1.
        theta1 = theta_list[1]
        vB_t1 = self.robot_B.velocity_at_theta(theta1)
        # Assume desired velocity is safe.
        vA_avoid1 = vA_des1

        A_t2 = A_t1 + vA_avoid1 * 1
        A_positions.append(A_t2)
        # At t2, compute new VO parameters.
        D2 = B_positions[2] - A_t2
        half_angle2 = self.vo.half_angle(A_t2, B_positions[2])
        vA_des2 = self.robot_A.speed * (D2 / np.linalg.norm(D2))
        theta2 = theta_list[2]
        vB_t2 = self.robot_B.velocity_at_theta(theta2)
        vA_avoid2 = vA_des2

        # Plot static trajectories.
        fig, ax = plt.subplots(figsize=(8, 6))
        A_positions = np.array(A_positions)
        B_positions = np.array(B_positions)

        ax.plot(A_positions[:, 0], A_positions[:, 1], "bo-", label="Robot A")
        ax.plot(B_positions[:, 0], B_positions[:, 1], "ro-", label="Robot B")

        # Plot robot B's full circular path.
        theta_vals = np.linspace(0, 2 * np.pi, 200)
        circle_B = self.robot_B.center.reshape(2, 1) + self.robot_B.radius * np.array(
            [np.cos(theta_vals), np.sin(theta_vals)]
        )
        ax.plot(circle_B[0, :], circle_B[1, :], "r--", alpha=0.5, label="B's path")

        # Plot VO cones at t0, t1, t2.
        self.vo.plot_vo(ax, self.robot_A.position, B_positions[0], half_angle0)
        self.vo.plot_vo(ax, A_positions[1], B_positions[1], half_angle1)
        self.vo.plot_vo(ax, A_positions[2], B_positions[2], half_angle2)

        # Plot velocity vectors for Robot A.
        ax.quiver(
            self.robot_A.position[0],
            self.robot_A.position[1],
            vA_avoid0[0],
            vA_avoid0[1],
            color="blue",
            angles="xy",
            scale_units="xy",
            scale=1,
            label="A's v at t0",
        )
        ax.quiver(
            A_positions[1][0],
            A_positions[1][1],
            vA_avoid1[0],
            vA_avoid1[1],
            color="green",
            angles="xy",
            scale_units="xy",
            scale=1,
            label="A's v at t1",
        )
        ax.quiver(
            A_positions[2][0],
            A_positions[2][1],
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
        ax.set_title("Static Plot: Trajectories and Velocity Obstacles")
        ax.legend()
        ax.axis("equal")
        ax.grid(True)
        plt.show()


class DynamicPlot:
    """Class to animate the simulation over time."""

    def __init__(self, total_time=20, dt=0.1):
        # Simulation parameters.
        self.total_time = total_time
        self.dt = dt
        self.times = np.arange(0, total_time, dt)
        self.radius_A = 1
        self.radius_B = 2
        self.safety_radius = self.radius_A + self.radius_B

        self.robot_A = Robot(position=[0, 0], radius=self.radius_A, speed=2.0)
        self.robot_B = Robot(
            position=[10, 0],
            radius=self.radius_B,
            speed=1.5,
            is_moving_in_circle=True,
            center=[10, 0],
        )
        self.vo = VelocityObstacle(self.safety_radius)

        # For dynamic simulation, we will store the evolving positions.
        self.A_positions = [self.robot_A.position.copy()]
        # Start time index 0.
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
        vA_des = self.robot_A.speed * D / np.linalg.norm(D)
        half_angle = self.vo.half_angle(A_pos, B_pos)
        # Compute relative desired velocity
        v_rel = vA_des - vB
        rel_angle = np.arctan2(v_rel[1], v_rel[0])
        # Check if the difference between line-of-sight and relative velocity is within half_angle.
        if abs(rel_angle - angle_D) < half_angle:
            # Adjust: choose the boundary (here, add half_angle).
            rel_angle_boundary = angle_D + half_angle
            v_rel_boundary = self.robot_A.speed * np.array(
                [np.cos(rel_angle_boundary), np.sin(rel_angle_boundary)]
            )
            vA = v_rel_boundary + vB
        else:
            vA = vA_des
        return vA

    def update(self, frame, scat_A, scat_B, quiv_A):
        # Update simulation time.
        self.current_time += self.dt

        # Update Robot B position on circle.
        theta = np.pi / 2 - self.robot_B.omega * self.current_time
        B_pos = self.robot_B.circular_position(theta)
        vB = self.robot_B.velocity_at_theta(theta)

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
        ax.set_title("Dynamic Plot: Simulation of Robots A and B")
        ax.axis("equal")
        ax.grid(True)

        # Plot Robot B's circular path.
        theta_vals = np.linspace(0, 2 * np.pi, 200)
        circle_B = self.robot_B.center.reshape(2, 1) + self.robot_B.radius * np.array(
            [np.cos(theta_vals), np.sin(theta_vals)]
        )
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
        B0 = self.robot_B.circular_position(theta0)
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

        # Use FuncAnimation for dynamic update.
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
    # Create and show the static plot.
    print("Displaying Static Plot...")
    sp = StaticPlot(t_values=[0, 10, 15])
    sp.plot()

    # Now run the dynamic simulation.
    print("Running Dynamic Simulation...")
    dp = DynamicPlot(total_time=20, dt=0.1)
    dp.run()


if __name__ == "__main__":
    main()
