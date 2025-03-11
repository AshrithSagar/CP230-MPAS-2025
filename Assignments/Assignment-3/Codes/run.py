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
    cc.compute()
    cc.plot()


if __name__ == "__main__":
    main()
