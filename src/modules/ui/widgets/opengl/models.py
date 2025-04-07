from pyqtgraph import Transform3D, Vector
from pyqtgraph.opengl import GLViewWidget, GLBoxItem
from PySide6.QtGui import QVector3D

from modules.ui.presets import MColors
from modules.ui.widgets.opengl.helpers import BasisVectors3D
from modules.ui.widgets.opengl.shapes import OpaqueCylinder


class Car3D:
    """Simple 3D car visualization"""

    def __init__(
        self,
        parent_widget: GLViewWidget,
        position=Vector(0, 0, 0),
        heading_deg=0.0,
        steering_angle_deg=0.0,
    ):
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
        camera_position = Vector(0.185, 0, 0.135)


        self.wheel_radius = wheel_radius
        self.wheel_width = wheel_width

        self.car_origin = BasisVectors3D(parent_widget=parent_widget, name="V", size=0.025)
        self.car_origin_offset = Vector(0, 0, 0)

        self.camera_origin = BasisVectors3D(parent_widget=parent_widget, name="C", size=0.025)
        self.camera_origin_offset = camera_position

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

        self._add_items_to_widget()

        self.x = position.x()
        self.y = position.y()
        self.heading_deg = 0
        self.steering_angle_deg = 0
        self.update_position(self.x, self.y, self.heading_deg, self.steering_angle_deg)

    def _add_items_to_widget(self):
        """Add all car components to the widget."""
        self.parent_widget.addItem(self.body)
        for wheel in self.wheels:
            self.parent_widget.addItem(wheel)

    def on_camera_change(self, opts: dict):
        self.car_origin.scale_font_by_camera_position(distance=opts["distance"], center=opts["center"])
        self.camera_origin.scale_font_by_camera_position(distance=opts["distance"], center=opts["center"])

    def update_position(self, x, y, heading_deg, steering_angle_deg):
        """Update the car's position and orientation."""
        self.x = x
        self.y = y
        self.heading_deg = heading_deg
        self.steering_angle_deg = steering_angle_deg

        t_body = Transform3D()
        t_body.translate(x, y, 0)
        t_body.rotate(heading_deg, 0, 0, 1)
        t_body.translate(self.body_offset)
        self.body.setTransform(t_body)

        t_car_origin = Transform3D()
        t_car_origin.translate(x, y, 0)
        t_car_origin.rotate(heading_deg, 0, 0, 1)
        t_car_origin.translate(self.car_origin_offset)
        self.car_origin.transform(t_car_origin)

        t_camera_origin = Transform3D()
        t_camera_origin.translate(x, y, 0)
        t_camera_origin.rotate(heading_deg, 0, 0, 1)
        t_camera_origin.translate(self.camera_origin_offset)
        self.camera_origin.transform(t_camera_origin)

        for i, wheel in enumerate(self.wheels):
            t_wheel = Transform3D()
            t_wheel.translate(x, y, 0)
            t_wheel.rotate(heading_deg, 0, 0, 1)
            t_wheel.translate(self.wheel_offsets[i])
            if i < 2:
                t_wheel.translate(Vector(0, -self.wheel_width / 2, 0))
                t_wheel.rotate(steering_angle_deg, 0, 0, 1)
                t_wheel.translate(Vector(0, self.wheel_width / 2, 0))
            t_wheel.translate(0, 0, self.wheel_radius)
            t_wheel.rotate(90, 1, 0, 0)
            wheel.setTransform(t_wheel)

