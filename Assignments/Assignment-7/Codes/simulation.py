"""
simulation.py \n
Simluate the velocity obstacle calculations
"""

from utils import *


def main():
    size = (600, 600)
    scene = Scene(time_step=0.02, sub_steps=1e3, size=size, scale=True)

    robotA = Robot(radius=5, size=size)
    robotB = Robot(radius=10, size=size)

    voAB = VelocityObstacle(robotA, robotB)

    scene.add_bodies([robotA, robotB])
    scene.add_vos([voAB])

    scene.render(framerate=240, record=False)


if __name__ == "__main__":
    main()
