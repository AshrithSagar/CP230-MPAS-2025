"""
collision_cone.py
Collision cone class
"""

import matplotlib.pyplot as plt
import numpy as np


class CollisionCone:
    def __init__(self, velocity_A, velocity_B, separation, theta0, alpha_B0):
        self.velocity_A = velocity_A
        self.velocity_B = velocity_B
        self.separation = separation  # Initial separation distance
        self.theta0 = theta0  # Initial angle between A and B
        self.alpha_B0 = alpha_B0  # Initial angle of B’s velocity

    def compute_collision_cone(self):
        """Compute the collision cone angles alpha_A1 and alpha_A2."""
        gamma = self.velocity_A / self.velocity_B  # Velocity ratio
        alpha_A1, alpha_A2 = self._calculate_cone_bounds(gamma)
        return alpha_A1, alpha_A2

    def _calculate_cone_bounds(self, gamma):
        """Private method to calculate the bounds of the collision cone."""
        delta = np.arctan2(
            self.separation * np.sin(self.theta0),
            self.separation * np.cos(self.theta0) - gamma,
        )
        alpha_A1 = self.alpha_B0 + delta
        alpha_A2 = self.alpha_B0 - delta
        return alpha_A1, alpha_A2

    def analyze_gamma_values(self, gamma_values):
        """Analyze different values of gamma to determine collision cone existence."""
        results = {}
        for gamma in gamma_values:
            alpha_A1, alpha_A2 = self._calculate_cone_bounds(gamma)
            exists = alpha_A1 is not None and alpha_A2 is not None
            results[gamma] = {
                "alpha_A1": alpha_A1,
                "alpha_A2": alpha_A2,
                "collision_cone_exists": exists,
            }
        return results

    def plot_collision_cone(self, gamma_values):
        """Visualize the collision cone as a function of gamma."""
        alpha_A1_vals = []
        alpha_A2_vals = []
        for gamma in gamma_values:
            alpha_A1, alpha_A2 = self._calculate_cone_bounds(gamma)
            alpha_A1_vals.append(alpha_A1)
            alpha_A2_vals.append(alpha_A2)

        plt.figure(figsize=(8, 5))
        plt.plot(gamma_values, alpha_A1_vals, label="Alpha A1", linestyle="--")
        plt.plot(gamma_values, alpha_A2_vals, label="Alpha A2", linestyle="--")
        plt.fill_between(
            gamma_values,
            alpha_A1_vals,
            alpha_A2_vals,
            color="gray",
            alpha=0.3,
            label="Collision Cone",
        )
        plt.xlabel("Gamma (velocity_A / velocity_B)")
        plt.ylabel("Alpha A")
        plt.title("Collision Cone as a Function of Gamma")
        plt.legend()
        plt.grid()
        plt.show()
