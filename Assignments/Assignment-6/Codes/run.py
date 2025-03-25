"""
run.py \n
Run the velocity obstacle calculation and plots
"""

from utils import *


def main():
    scene = Scene(time_step=0.1, sub_steps=100, scale=True)

    robotA = Robot(radius=3)
    robotB = Robot(radius=4)
    robotC = Robot(radius=5)

    scene.add_bodies([robotA, robotB, robotC])
    scene.render(framerate=60)


if __name__ == "__main__":
    main()
