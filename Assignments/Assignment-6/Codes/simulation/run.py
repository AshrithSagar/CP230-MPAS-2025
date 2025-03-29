"""
simulation/run.py \n
Run the velocity obstacle calculation and plots
"""

from utils import *


def main():
    scene = Scene(time_step=0.01, sub_steps=1e3, scale=True)

    robotA = Robot(radius=3)
    robotB = Robot(radius=4)
    robotC = Robot(radius=5)

    voAB = VelocityObstacle(robotA, robotB)
    voBC = VelocityObstacle(robotB, robotC)
    voCA = VelocityObstacle(robotC, robotA)

    scene.add_bodies([robotA, robotB, robotC])
    scene.add_vos([voAB, voBC, voCA])

    scene.render(framerate=240, record=False)


if __name__ == "__main__":
    main()
