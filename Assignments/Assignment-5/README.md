# Assignment-5

Group of 2 students

Due date: 24th March 2025, 12:00 PM

---

2D motion planning in the $x$-$z$ plane (gravity is acting vertically downwards)

Will the robot reach it's goal?

![Tasks](./Files/tasks.jpeg)

## Task 1

Model a point-sized robot of unit mass dropped from a certain height.
It falls under gravity acting in negative $z$ direction and on impact with the ground; normal reaction and impact force act on it.
Choose $\epsilon$ as per convenience.
Also, consider an attractive force acting on it due to a stationary goal far away in the $x$-direction.
The robot is built for a maximum horizontal speed capability of $v_{\max}$.

## Task 2

Consider a static obstacle with a virtual periphery.
A repulsive field should act on the robot when it breaches the periphery.
Choose the obstacle geometry wisely, as the robot may get trapped in a local minima.

## Task 3

Consider a point-sized moving obstacle, moving head-on towards the robot.
Equip the robot with a virtual periphery to detect the collision threat from the moving obstacle.
Design the repulsive field for the robot to avoid collision.

## Task 4

Consider a tunnel at some height.
After avoiding the moving obstacle, the robot should enter and pass through the tunnel without touching the tunnel’s sidewalls.

## Task 5

Once the robot exits the tunnel, consider the stationary goal point starts moving with velocity $v_d$ in the $x$-direction.
Note that $v_{\max} > v_d$.
