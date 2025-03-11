"""
run.py
Collision cone analysis
"""

import numpy as np
from collision_cone import CollisionCone


def main():
    cc = CollisionCone(
        vA=10.0,
        vB=5.0,
        r0=10.0,
        th=np.radians(30),
        alphB=np.radians(120),
        rB=2.0,
    )
    alpha_A = cc.compute()
    for func, desc in [(min, "Minimum"), (max, "Maximum")]:
        value = func(alpha_A).round(2)
        print(f"{desc} alpha_A: {value} radians = {np.degrees(value).round(2)} degrees")
    cc.plot(gamma=np.linspace(-50, 50, 100))


if __name__ == "__main__":
    main()
