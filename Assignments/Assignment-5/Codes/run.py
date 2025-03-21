"""
run.py
Run the tasks
"""

from utils import Obstacle, PointRobot, Scene


def main():
    scene = Scene(display_size=(800, 600), elasticity=1.0, dt=0.2, steps=10)

    robot = PointRobot(position=(400, 300), velocity=(0, 0), mass=1, vmax=10)
    obstacle = Obstacle(position=(200, 400))

    scene.add_bodies([robot, obstacle])
    scene.render()


if __name__ == "__main__":
    main()
