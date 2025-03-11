"""
collision_cone.py
Collision cone class
"""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class CollisionCone:
    def __init__(
        self, vA: float, vB: float, r0: float, th: float, alphB: float, rB: float
    ) -> None:
        self.vA = vA  # Velocity of A
        self.vB = vB  # Velocity of B
        self.r0 = r0  # Initial r0 distance
        self.th = th  # Inclination of the r0 vector
        self.alphB = alphB  # Initial angle of B’s velocity
        self.rB = rB  # Radius of B
        self.g = vA / vB  # Speed ratio
        self.alpha_A: List[float] = []

    def compute(self) -> List[float]:
        self.alpha_A = []
        for alphA in np.linspace(0, 2 * np.pi, 20000):
            LHS = (np.sin(self.alphB - self.th) - self.g * np.sin(alphA - self.th)) ** 2
            LHS *= self.r0**2 - self.rB**2
            RHS = (np.cos(self.alphB - self.th) - self.g * np.cos(alphA - self.th)) ** 2
            RHS *= self.rB**2
            if LHS < RHS:
                self.alpha_A.append(alphA)
        return self.alpha_A

    def plot(self) -> None:
        gamma = np.linspace(0, 2, 100)
        min_alphA = np.ones_like(gamma) * np.inf
        max_alphA = np.ones_like(gamma) * -np.inf
        for i, g in enumerate(gamma):
            self.g = g
            alpha_A = self.compute()
            if alpha_A:
                min_alphA[i] = min(alpha_A)
                max_alphA[i] = max(alpha_A)
        plt.plot(gamma, min_alphA, color="black")
        plt.plot(gamma, max_alphA, color="black")
        plt.fill_between(gamma, min_alphA, max_alphA, color="gray", alpha=0.5)
        plt.xlabel("$\gamma$")
        plt.ylabel("$\\alpha_A$")
        plt.title("$\\alpha_A$ vs. $\gamma$")
        plt.xlim(min(gamma), max(gamma))
        plt.ylim(0, 2 * np.pi)
        plt.legend()
        plt.show()
