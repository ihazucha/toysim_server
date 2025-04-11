from pyqtgraph import Transform3D, Vector
from pyqtgraph.opengl import GLViewWidget, GLBoxItem

from modules.ui.presets import MColors, Colors
from modules.ui.widgets.opengl.helpers import ReferenceFrame
from modules.ui.widgets.opengl.shapes import OpaqueCylinder
from PySide6.QtGui import QColor

class CarProps:
    CHASSIS_LENGTH = 0.255
    CHASSIS_WIDTH = 0.155
    CHASSIS_HEIGHT = 0.005
    WHEELBASE = 0.185
    TRACK = 0.155
    WHEEL_RADIUS = 0.03
    WHEEL_WIDTH = 0.02
    CAMERA_POS = Vector(0.185, 0, 0.135)
    CAMERA_ROT = Vector(0, 0, 0)


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

        self.car_rf = ReferenceFrame(parent_widget, name="Vehicle", size=0.025)
        self._car_rf_offset = Vector(0, 0, 0)

        self.camera_rf = ReferenceFrame(parent_widget, name="Camera", size=0.025)
        self._camera_rf_offset = CarProps.CAMERA_POS

        dimensions = Vector(
            CarProps.CHASSIS_LENGTH, CarProps.CHASSIS_WIDTH, CarProps.CHASSIS_HEIGHT
        )
        self.body = GLBoxItem(size=dimensions, color=Colors.ON_ACCENT)
        self.body.lineplot.setData(width=2, antialias=True)
        self._body_offset = Vector(-0.01, -CarProps.CHASSIS_WIDTH / 2, CarProps.WHEEL_RADIUS)

        # FL, FR, RL, RR
        self._wheel_offsets = [
            Vector(CarProps.WHEELBASE, CarProps.CHASSIS_WIDTH / 2 + CarProps.WHEEL_WIDTH, 0),
            Vector(CarProps.WHEELBASE, -CarProps.CHASSIS_WIDTH / 2, 0),
            Vector(0, CarProps.CHASSIS_WIDTH / 2 + CarProps.WHEEL_WIDTH, 0),
            Vector(0, -CarProps.CHASSIS_WIDTH / 2, 0),
        ]
        self.wheels = []
        for _ in self._wheel_offsets:
            wheel = OpaqueCylinder(
                radius=CarProps.WHEEL_RADIUS, height=CarProps.WHEEL_WIDTH, color=QColor(Colors.ON_ACCENT),
            )
            self.wheels.append(wheel)

        self._add_items_to_widget()

        self.position = position
        self.heading_deg = heading_deg
        self.steering_angle_deg = steering_angle_deg
        self.update(self.position.x(), self.position.y(), self.heading_deg, self.steering_angle_deg)

    def _add_items_to_widget(self):
        self.parent_widget.addItem(self.body)
        for wheel in self.wheels:
            self.parent_widget.addItem(wheel)

    def update(self, x, y, heading_deg, steering_angle_deg):
        self.position.setX(x)
        self.position.setY(y)
        self.heading_deg = heading_deg
        self.steering_angle_deg = steering_angle_deg

        for item, offset in [
            (self.body, self._body_offset),
            (self.car_rf, self._car_rf_offset),
            (self.camera_rf, self._camera_rf_offset),
        ]:
            t = Transform3D()
            t.translate(x, y, 0)
            t.rotate(heading_deg, 0, 0, 1)
            t.translate(offset)
            item.setTransform(t)

        for i, wheel in enumerate(self.wheels):
            t_wheel = Transform3D()
            t_wheel.translate(x, y, 0)
            t_wheel.rotate(heading_deg, 0, 0, 1)
            t_wheel.translate(self._wheel_offsets[i])
            if i < 2:
                t_wheel.translate(Vector(0, -CarProps.WHEEL_WIDTH / 2, 0))
                t_wheel.rotate(steering_angle_deg, 0, 0, 1)
                t_wheel.translate(Vector(0, CarProps.WHEEL_WIDTH / 2, 0))
            t_wheel.translate(0, 0, CarProps.WHEEL_RADIUS)
            t_wheel.rotate(90, 1, 0, 0)
            wheel.setTransform(t_wheel)
