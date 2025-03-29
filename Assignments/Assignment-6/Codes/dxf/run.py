"""
dxf/run.py \n
Draw the velocity obstacles using ezdxf CAD
"""

from ezdxf.math import ConstructionCircle, Vec2
from utils import *


def main():
    dxf = DXF()

    origin = Vec2(0, 0)  # robotA
    robotB = ConstructionCircle(center=Vec2(2, 10), radius=2)
    robotC = ConstructionCircle(center=Vec2(10, 6), radius=1.5)
    dxf.draw_point(point=origin)
    dxf.draw_circle(circle=robotB)
    dxf.draw_circle(circle=robotC)

    dxf.draw_tangents_to_circle_from_point(circle=robotB, point=origin)

    dxf.save_as_png(filename="dxf.png")


if __name__ == "__main__":
    main()
