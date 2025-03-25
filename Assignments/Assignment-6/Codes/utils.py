"""
utils.py \n
Utility classes for velocity obstacle calculations.
"""

import numpy as np

np.random.seed(25)


class Robot:
    def __init__(self):
        self.position = np.random.uniform(0, 200, size=2)
        speed = np.random.uniform(10, 50)
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = speed * np.array([np.cos(angle), np.sin(angle)])
        self.radius = np.random.uniform(1, 5)

    def __repr__(self):
        return f"Robot(position={self.position}, velocity={self.velocity}, radius={self.radius})"
