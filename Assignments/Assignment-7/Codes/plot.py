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

    def plot_vo(self, ax, A, B, half_angle, color="gray", alpha=0.4):
        """Plot the two boundary rays of the VO cone at point A with target B."""
        lines = []
        D = B - A
        d = np.linalg.norm(D)
        angle_center = np.arctan2(D[1], D[0])
        angles = [angle_center - half_angle, angle_center + half_angle]
        for a in angles:
            # Extend rays for visualization (1.5*d)
            ray = A.reshape(2, 1) + np.array([[np.cos(a)], [np.sin(a)]]) * d * 1.5
            (line,) = ax.plot(
                [A[0], ray[0, 0]],
                [A[1], ray[1, 0]],
                color=color,
                linestyle="--",
                alpha=alpha,
            )
            lines.append(line)
        return lines


class DynamicPlot:
    """
    Class to animate the simulation over time. \n
    Initially, Robot A moves head-on toward Robot B.
    At each time in avoid_times, a new avoidance velocity is chosen from the avoidance set and held until the next avoid time.
    The VO cone is re-plotted at each time step.
    """

    def __init__(
        self,
        bodies: Tuple[Robot, Robot],
        avoid_times: List[float],
        total_time: float = 20,
        time_step: float = 0.1,
        max_vo_history: int = 1,
    ):
        self.robotA, self.robotB = bodies
        self.avoid_times = sorted(avoid_times)
        self.total_time = total_time
        self.dt = time_step
        self.times = np.arange(0, total_time, time_step)
        self.vo = VelocityObstacle(self.robotA, self.robotB)
        self.A_positions = [self.robotA.position.copy()]
        self.current_time = 0
        self.next_avoid_index = 0
        self.current_vA = None
        self.vo_artists = []
        self.max_vo_history = max_vo_history

    def compute_avoidance_velocity(self, A_pos, B_pos, vB):
        D = B_pos - A_pos
        if np.linalg.norm(D) == 0:
            return np.zeros(2)
        angle_D = np.arctan2(D[1], D[0])
        vA_des = self.robotA.speed * D / np.linalg.norm(D)
        half_angle = self.vo.half_angle(A_pos, B_pos)
        v_rel = vA_des - vB
        rel_angle = np.arctan2(v_rel[1], v_rel[0])
        if abs(rel_angle - angle_D) < half_angle:
            rel_angle_boundary = angle_D + half_angle
            v_rel_boundary = self.robotA.speed * np.array(
                [np.cos(rel_angle_boundary), np.sin(rel_angle_boundary)]
            )
            vA = v_rel_boundary + vB
        else:
            vA = vA_des
        return vA

    def update(self, frame, scat_A, scat_B, quiv_A):
        self.current_time += self.dt
        theta = np.pi / 2 - self.robotB.omega * self.current_time
        B_pos = self.robotB.circular_position(theta)
        vB = self.robotB.velocity_at_theta(theta)

        if (
            self.next_avoid_index < len(self.avoid_times)
            and self.current_time >= self.avoid_times[self.next_avoid_index]
        ):
            A_current = self.A_positions[-1]
            self.current_vA = self.compute_avoidance_velocity(A_current, B_pos, vB)
            self.next_avoid_index += 1
        elif self.current_vA is None:
            A_current = self.A_positions[-1]
            D = B_pos - A_current
            self.current_vA = self.robotA.speed * D / np.linalg.norm(D)

        A_current = self.A_positions[-1]
        new_A_pos = A_current + self.current_vA * self.dt
        self.A_positions.append(new_A_pos)

        scat_A.set_offsets(new_A_pos.reshape(1, -1))
        scat_B.set_offsets(B_pos.reshape(1, -1))
        quiv_A.set_offsets(new_A_pos.reshape(1, -1))
        quiv_A.set_UVC(self.current_vA[0], self.current_vA[1])

        if len(self.vo_artists) >= self.max_vo_history:
            for artist in self.vo_artists.pop(0):
                artist.remove()
        half_angle = self.vo.half_angle(new_A_pos, B_pos)
        self.vo_artists.append(self.vo.plot_vo(self.ax, new_A_pos, B_pos, half_angle))

        return scat_A, scat_B, quiv_A

    def run(self, record: bool = False, record_params={}):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Dynamic Simulation with Velocity Obstacles")
        self.ax.axis("equal")
        self.ax.grid(True)

        theta_vals = np.linspace(0, 2 * np.pi, 200)
        circle_B = self.robotB.circling_around.reshape(
            2, 1
        ) + self.robotB.radius * np.array([np.cos(theta_vals), np.sin(theta_vals)])
        self.ax.plot(circle_B[0, :], circle_B[1, :], "r--", alpha=0.5, label="B's path")

        scat_A = self.ax.scatter(
            self.A_positions[0][0],
            self.A_positions[0][1],
            color="blue",
            s=50,
            label="Robot A",
        )
        theta0 = np.pi / 2
        B0 = self.robotB.circular_position(theta0)
        scat_B = self.ax.scatter(B0[0], B0[1], color="red", s=50, label="Robot B")
        quiv_A = self.ax.quiver(
            self.A_positions[0][0],
            self.A_positions[0][1],
            0,
            0,
            color="blue",
            scale=1,
            scale_units="xy",
        )

        self.ax.legend()
        anim = FuncAnimation(
            self.fig,
            self.update,
            fargs=(scat_A, scat_B, quiv_A),
            frames=len(self.times),
            interval=self.dt * 1000,
            blit=False,
        )

        if record:
            file = record_params["filename"]
            fps = record_params["fps"]
            anim.save(file, fps=fps, extra_args=["-vcodec", "libx264"])
            print(f"Animation saved to {file}")
        else:
            plt.show()


def main():
    robotA = Robot(position=(0, 0), radius=1, speed=2)
    robotB = Robot(position=(10, 2), radius=5, speed=1.5, circling_around=(10, 0))

    scene = DynamicPlot(
        bodies=[robotA, robotB],
        avoid_times=[3, 5, 8],
        total_time=12,
        time_step=0.1,
    )
    scene.run(
        record=True,
        record_params={
            "filename": "animation.mp4",
            "fps": 20,
        },
    )


if __name__ == "__main__":
    main()
