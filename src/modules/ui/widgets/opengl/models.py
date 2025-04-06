from pyqtgraph import Transform3D, Vector
from pyqtgraph.opengl import GLViewWidget, GLBoxItem
from PySide6.QtGui import QVector3D

from modules.ui.presets import MColors
from modules.ui.widgets.opengl.helpers import BasisVectors3D
from modules.ui.widgets.opengl.shapes import OpaqueCylinder


class Car3D:
    """Simple 3D car visualization"""

    def __init__(self, parent_widget: GLViewWidget, position=Vector(0, 0, 0)):
        """Create a simple car model starting at origin."""
        self.parent_widget = parent_widget

        # TODO: refactor into datalink
        chassis_len = 0.255
        chassis_width = 0.155
        chassis_height = 0.005
        wheelbase = 0.185
        track = 0.155
        wheel_radius = 0.03
        wheel_width = 0.02
        self.wheel_radius = wheel_radius
        self.wheel_width = wheel_width

        self.origin = BasisVectors3D(parent_widget=parent_widget, name="V")
        self.origin_offset = Vector(0, 0, 0)

        self.body = GLBoxItem(
            size=Vector(chassis_len, chassis_width, chassis_height), color=MColors.RED_TRANS
        )
        self.body_offset = Vector(0, -chassis_width / 2, wheel_radius)

        self.wheels = []
        self.wheel_offsets = []

        # FL, FR, RL, RR
        wheel_positions = [
            (wheelbase, chassis_width / 2 + wheel_width, -wheel_radius),
            (wheelbase, -chassis_width / 2, -wheel_radius),
            (0, chassis_width / 2 + wheel_width, -wheel_radius),
            (0, -chassis_width / 2, -wheel_radius),
        ]

        for i, (wx, wy, wz) in enumerate(wheel_positions):
            wheel = OpaqueCylinder(radius=wheel_radius, height=wheel_width)
            self.wheels.append(wheel)
            self.wheel_offsets.append(QVector3D(wx, wy, wz + wheel_radius))

        self.add_to_widget(parent_widget)

        self.x = 0
        self.y = 0
        self.heading = 0
        self.update_position(0, 0, 0, 0)

    def add_to_widget(self, widget):
        """Add all car components to the widget."""
        widget.addItem(self.body)
        for wheel in self.wheels:
            widget.addItem(wheel)

    def update_position(self, x, y, heading_deg, steering_angle_deg):
        """Update the car's position and orientation."""
        self.x = x
        self.y = y
        self.heading = heading_deg

        transform_body = Transform3D()
        transform_body.translate(x, y, 0)
        transform_body.rotate(heading_deg, 0, 0, 1)
        transform_body.translate(self.body_offset)
        self.body.setTransform(transform_body)
        
        transform_origin = Transform3D()
        transform_origin.translate(x, y, 0)
        transform_origin.rotate(heading_deg, 0, 0, 1)
        transform_origin.translate(self.origin_offset)
        self.origin.transform(transform_origin)

        for i, wheel in enumerate(self.wheels):
            transform_wheel = Transform3D()
            transform_wheel.translate(x, y, 0)
            transform_wheel.rotate(heading_deg, 0, 0, 1)
            transform_wheel.translate(self.wheel_offsets[i])
            if i < 2:
                transform_wheel.translate(Vector(0, -self.wheel_width/2, 0))
                transform_wheel.rotate(steering_angle_deg, 0, 0, 1)
                transform_wheel.translate(Vector(0, self.wheel_width/2, 0))
            transform_wheel.translate(0, 0, self.wheel_radius)
            transform_wheel.rotate(90, 1, 0, 0)

            wheel.setTransform(transform_wheel)

