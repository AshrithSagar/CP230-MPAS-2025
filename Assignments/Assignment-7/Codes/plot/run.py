"""
draw/run.py \n
Draw the velocity obstacle calculation and plots
"""

import matplotlib.pyplot as plt
import numpy as np


def posB(theta):
    return center_B + radius_B * np.array([np.cos(theta), np.sin(theta)])


def vo_half_angle(A, B):
    """VO half-angle from A to B"""
    D = B - A
    d = np.linalg.norm(D)
    return np.arcsin(safety_radius / d) if d > safety_radius else np.pi / 2


# Parameters
radius_A = 1
radius_B = 2
safety_radius = radius_A + radius_B
t0, t1, t2 = 0, 10, 15

# Robot A
speed_A_des = 2.0
A_t0 = np.array([0, 0])

# Robot B
center_B = np.array([10, 0])
speed_B = 1.5
omega = speed_B / radius_B
theta0 = np.pi / 2
B_t0 = posB(theta0)
B_t1 = posB(theta0 - omega * t1)
B_t2 = posB(theta0 - omega * t2)

# At t0:
D0 = B_t0 - A_t0
angle_D0 = np.arctan2(D0[1], D0[0])
half_angle0 = vo_half_angle(A_t0, B_t0)
# Desired velocity (collision course)
vA_des0 = speed_A_des * D0 / np.linalg.norm(D0)
# Relative desired velocity (would be vA_des0 - vB; vB at t0)
vB_t0 = np.array([-1.5, 0])
# Check if inside VO: Purposely compute a velocity on the boundary.
# Choose the boundary that is above the line-of-sight:
rel_angle_boundary = angle_D0 + half_angle0
# Relative velocity needed (magnitude = speed_A_des)
v_rel_boundary = speed_A_des * np.array(
    [np.cos(rel_angle_boundary), np.sin(rel_angle_boundary)]
)
vA_avoid0 = v_rel_boundary + vB_t0  # A's avoidance velocity at t0

# At t1:
A_t1 = A_t0 + vA_avoid0 * 1  # Assume constant velocity for 1 sec for illustration
D1 = B_t1 - A_t1
half_angle1 = vo_half_angle(A_t1, B_t1)
vA_des1 = speed_A_des * (D1 / np.linalg.norm(D1))
theta1 = theta0 - omega * t1  # B's velocity at t1:
# Tangent: derivative of B position is (-radius*sin(theta), radius*cos(theta))
vB_t1 = omega * radius_B * np.array([-np.sin(theta1), np.cos(theta1)])
v_rel1 = vA_des1 - vB_t1
# Assume that v_rel1 lies outside the cone, so A can take desired velocity.
vA_avoid1 = vA_des1


# At t2:
A_t2 = A_t1 + vA_avoid1 * 1
D2 = B_t2 - A_t2
half_angle2 = vo_half_angle(A_t2, B_t2)
vA_des2 = speed_A_des * (D2 / np.linalg.norm(D2))
theta2 = theta0 - omega * t2
vB_t2 = omega * radius_B * np.array([-np.sin(theta2), np.cos(theta2)])
v_rel2 = vA_des2 - vB_t2
vA_avoid2 = vA_des2  # Assume safe

## Plots
fig, ax = plt.subplots(figsize=(8, 6))

# Robot trajectories
ax.plot(
    [A_t0[0], A_t1[0], A_t2[0]], [A_t0[1], A_t1[1], A_t2[1]], "bo-", label="Robot A"
)
ax.plot(
    [B_t0[0], B_t1[0], B_t2[0]], [B_t0[1], B_t1[1], B_t2[1]], "ro-", label="Robot B"
)

# Robot B's path
theta_vals = np.linspace(0, 2 * np.pi, 200)
circle_B = center_B.reshape(2, 1) + radius_B * np.array(
    [np.cos(theta_vals), np.sin(theta_vals)]
)
ax.plot(circle_B[0, :], circle_B[1, :], "r--", alpha=0.5, label="B's path")


# VO cones at t0, t1, t2 for robot A
def plot_vo(A, B, half_angle, color="gray"):
    D = B - A
    d = np.linalg.norm(D)
    angle_center = np.arctan2(D[1], D[0])
    angles = [angle_center - half_angle, angle_center + half_angle]
    for a in angles:
        # Ray is taken as 1.5 times the length
        ray = A.reshape(2, 1) + np.array([[np.cos(a)], [np.sin(a)]]) * d * 1.5
        ax.plot([A[0], ray[0, 0]], [A[1], ray[1, 0]], color=color, linestyle="--")


plot_vo(A_t0, B_t0, half_angle0, color="gray")
plot_vo(A_t1, B_t1, half_angle1, color="gray")
plot_vo(A_t2, B_t2, half_angle2, color="gray")

# Velocity vectors for A
ax.quiver(
    A_t0[0],
    A_t0[1],
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
