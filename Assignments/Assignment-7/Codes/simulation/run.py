"""
simulation/run.py \n
Run the velocity obstacle calculation and plots
"""

from utils import *


def main():
    scene = Scene(time_step=0.01, sub_steps=1e3, scale=True)

    robotA = Robot(radius=3)
    robotB = Robot(radius=4)

    voAB = VelocityObstacle(robotA, robotB)

    scene.add_bodies([robotA, robotB])
    scene.add_vos([voAB])

    scene.render(framerate=240, record=False)


if __name__ == "__main__":
    main()
