"""
run.py
Collision cone analysis
"""

import numpy as np
from collision_cone import CollisionCone


def main():
    cc = CollisionCone(
        velocity_A=10.0,
        velocity_B=5.0,
        separation=100.0,
        theta0=np.pi / 4,
        alpha_B0=np.pi / 6,
    )
    alpha_A1, alpha_A2 = cc.compute_collision_cone()
    print(f"Alpha A1: {alpha_A1:.2f} radians")
    print(f"Alpha A2: {alpha_A2:.2f} radians")
    gamma_values = np.linspace(0.1, 2.0, 50)
    results = cc.analyze_gamma_values(gamma_values)
    for gamma, result in results.items():
        print(f"Gamma: {gamma:.2f}")
        print(f"  Alpha A1: {result['alpha_A1']:.2f}")
        print(f"  Alpha A2: {result['alpha_A2']:.2f}")
        print(f"  Collision Cone Exists: {result['collision_cone_exists']}")
        print()
    cc.plot_collision_cone(gamma_values)


if __name__ == "__main__":
    main()
