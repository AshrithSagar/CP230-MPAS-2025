"""
dxf/utils.py \n
Utility classes for velocity obstacle calculations.
"""

import math
import tempfile

import ezdxf
from ezdxf import recover, units
from ezdxf.addons.drawing import matplotlib, properties
from ezdxf.math import ConstructionCircle, Vec2

properties.MODEL_SPACE_BG_COLOR = "#FFFFFF"  # White background


class DXF:
    """DXF class to handle DXF file operations."""

    def __init__(self, units=units.MM):
        self.doc = ezdxf.new()
        self.doc.units = units
        self.msp = self.doc.modelspace()

    def save_as_png(self, filename: str = "output.png"):
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=True) as temp_dxf:
            self.doc.saveas(temp_dxf.name)
            doc, auditor = recover.readfile(temp_dxf.name)
            if not auditor.has_errors:
                matplotlib.qsave(doc.modelspace(), filename)

    def draw_point(self, point: Vec2):
        self.msp.add_point(location=point)

    def draw_circle(self, circle: ConstructionCircle):
        self.msp.add_circle(center=circle.center, radius=circle.radius)

    def draw_tangents_to_circle_from_point(
        self, circle: ConstructionCircle, point: Vec2
    ):
        vector_cp: Vec2 = point - circle.center
        distance = vector_cp.magnitude
        if distance <= circle.radius:
            raise ValueError("Point must be outside the circle")

        tangent_distance = math.sqrt(distance**2 - circle.radius**2)
        angle = math.atan2(vector_cp.y, vector_cp.x)
        offset_angle = math.asin(circle.radius / distance)

        angle1 = angle + offset_angle
        angle2 = angle - offset_angle

        tangent1 = (
            circle.center + Vec2(math.cos(angle1), math.sin(angle1)) * circle.radius
        )
        tangent2 = (
            circle.center + Vec2(math.cos(angle2), math.sin(angle2)) * circle.radius
        )

        self.msp.add_line(start=point, end=tangent1)
        self.msp.add_line(start=point, end=tangent2)
