import sys

sys.path.append("C:/Users/ihazu/Desktop/projects/toysim_server/src")

import numpy as np
from typing import Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QQuaternion, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
)

from pyqtgraph.opengl import GLViewWidget, GLGridItem, GLLinePlotItem
from pyqtgraph import Transform3D, Vector

from modules.ui.presets import MColors
from modules.ui.widgets.opengl.shapes import OpaqueCube, customNormalColor, Cone

# Helpers
# -----------------------------------------------------------------------------


def quat2euler(quaternion: QQuaternion) -> Tuple[float, float, float]:
    w = quaternion.scalar()
    x = quaternion.x()
    y = quaternion.y()
    z = quaternion.z()

    # Roll (x)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def print_quat(quaternion: QQuaternion):
    roll, pitch, yaw = quat2euler(quaternion)
    roll_deg, pitch_deg, yaw_deg = np.rad2deg([roll, pitch, yaw])
    print(
        f"\rRoll: {roll_deg:.2f}°, Pitch: {pitch_deg:.2f}°, Yaw: {yaw_deg:.2f}°", end=""
    )

# Widgets
# -----------------------------------------------------------------------------


class IMUReferenceFrame:
    def __init__(
        self,
        parent_widget: GLViewWidget,
        size=0.025,
        width=3,
        alpha=255,
        show_cube=True,
    ):
        self.parent_widget = parent_widget
        self.size = size
        self.show_cube = show_cube
        self.width=width
        self.alpha=alpha

        self._add_axis_lines()
        if self.show_cube:
            self._add_origin_cube()

        self.setTransform(Transform3D())

    def get_position(self):
        return Vector(self.x.transform().matrix()[:3, 3])

    def get_transform(self):
        return self.x.transform()

    def setTransform(self, tr: Transform3D):
        for line in self.xyz_lines:
            line.setTransform(tr)

        if self.show_cube:
            tr.translate(self.origin_offset)
            self.origin.setTransform(tr)

    def _add_axis_lines(self):
        for axis_name, color, end_point in (
            ("x", QColor(MColors.RED), [self.size, 0, 0]),
            ("y", QColor(MColors.GREEN), [0, self.size, 0]),
            ("z", QColor(MColors.BLUE), [0, 0, self.size]),
        ):
            color.setAlpha(self.alpha)
            axis_line = GLLinePlotItem(
                pos=np.array([[0, 0, 0], end_point]), color=color, width=self.width, antialias=True
            )
            setattr(self, axis_name, axis_line)
            self.parent_widget.addItem(axis_line)

        self.xyz_lines = [self.x, self.y, self.z]

    def _add_origin_cube(self):
        radius = self.size / 4
        self.origin_offset = Vector(*[-radius / 2] * 3)
        self.origin = OpaqueCube(size=radius, shader=customNormalColor, glOptions="translucent")
        self.origin.setDepthValue(-1)
        self.parent_widget.addItem(self.origin)



class ArcItem:
    def __init__(
        self,
        parent_widget: GLViewWidget,
        segments=32,
        radius=1,
        width=1,
        color: QColor = MColors.WHITE,
        base_transform=Transform3D(),
    ):
        self.parent_widget = parent_widget
        self.segments = segments
        self.radius = radius
        self.base_transform = base_transform

        self.line = GLLinePlotItem(color=color, width=width, antialias=True)
        self.parent_widget.addItem(self.line)

    def update(self, start_vec: Vector, end_vec: Vector):
        start_unit_vec = start_vec.normalized()
        end_unit_vec = end_vec.normalized()

        dot_product = Vector.dotProduct(start_unit_vec, end_unit_vec)
        rotation_vec = Vector.crossProduct(start_unit_vec, end_unit_vec)

        self.arc_angle = np.arccos(dot_product)

        arc_points = np.zeros((self.segments + 1, 3))
        for i in range(self.segments + 1):
            arc_step = i / (self.segments)
            angle_step = self.arc_angle * arc_step

            tr = Transform3D()
            tr.rotate(np.degrees(angle_step), rotation_vec)
            
            rotated_vector = tr.map(start_unit_vec)
            scaled_vector = rotated_vector * self.radius
            arc_points[i] = scaled_vector.toTuple()

        self.line.setData(pos=np.array(arc_points))

    def setTransform(self, tr):
        self.line.setTransform(tr)


class IMU3D(GLViewWidget):
    INIT_OPTS = {
        "center": Vector(0, 0, 0.1),
        "distance": 1.5,
        "elevation": 30,
        "azimuth": 135,
    }

    def __init__(self):
        super().__init__()
        self.setBackgroundColor((10, 10, 10, 255))
        self.setCameraPosition(
            pos=self.INIT_OPTS["center"],
            distance=self.INIT_OPTS["distance"],
            elevation=self.INIT_OPTS["elevation"],
            azimuth=self.INIT_OPTS["azimuth"],
        )

        # Items
        self.world_rf = IMUReferenceFrame(parent_widget=self, size=0.5, alpha=128, show_cube=False)
        self.imu_rf = IMUReferenceFrame(parent_widget=self, size=0.25, width=5)
        
        self._add_grids()
        self._add_angle_arcs()
        self._add_accel_arrows()
        self._add_gyro_indicators()

    def _add_angle_arcs(self, arc_radius=0.25):
        self.angle_arcs = []
        for axis, color in (
            ("x", QColor(MColors.RED)),
            ("y", QColor(MColors.GREEN)),
            ("z", QColor(MColors.BLUE)),
        ):
            color.setAlpha(80)
            arc_item = ArcItem(parent_widget=self, color=color, width=2, radius=arc_radius)
            self.angle_arcs.append(arc_item)

    def _add_accel_arrows(self):
        self.accel_arrows = []
        for axis, color, offset in (
            ("x", QColor(MColors.RED), (90, 0, 1, 0)),
            ("y", QColor(MColors.GREEN), (-90, 1, 0, 0)),
            ("z", QColor(MColors.BLUE), (0, 0, 0, 1)),
        ):
            color.setAlpha(80)
            world_rf_axis = getattr(self.world_rf, axis)

            tr = Transform3D()
            tr.translate(*world_rf_axis.pos[1])
            tr.rotate(*offset)

            arrow = Cone(color=color, length=0.1, radius=(0.02, 0), base_transform=tr)

            self.accel_arrows.append(arrow)
            self.addItem(arrow)

    def _add_gyro_indicators(self, arc_radius=0.2):
        self.gyro_arcs = []
        self.gyro_arrows = []

        for axis, color, arrow_rotation, arc_offset in (
            ("x", QColor(MColors.RED), (90, 1, 0, 0), Vector(0, 0, arc_radius)),
            ("y", QColor(MColors.GREEN), (90, 0, 1, 0), Vector(0, 0, arc_radius)),
            ("z", QColor(MColors.BLUE), (90, 1, 0, 0), Vector(arc_radius, 0, 0)),
        ):
            color.setAlpha(128)

            arc_tr = Transform3D()
            arc_tr.translate(arc_offset)
            arc_item = ArcItem(
                parent_widget=self, color=color, width=3, radius=arc_radius, base_transform=arc_tr
            )
            self.gyro_arcs.append(arc_item)

            arrow_tr = Transform3D()
            arrow_tr.rotate(*arrow_rotation)
            arrow = Cone(color=color, length=0.1, radius=(0.02, 0), base_transform=arrow_tr)
            self.gyro_arrows.append(arrow)
            self.addItem(arrow)

    def update_data(self, quaternion: QQuaternion, accel: Vector, gyro: Vector):
        tr = Transform3D()
        tr.rotate(quaternion)
        self.imu_rf.setTransform(tr)

        for i in range(len(self.world_rf.xyz_lines)):
            self._update_angle_arc(i, tr)
            self._update_accel_arrow(self.accel_arrows[i], accel[i])
            self._update_gyro_arc(i, gyro[i])

    def _update_angle_arc(self, axis_index: str, tr: Transform3D):
        arc: ArcItem = self.angle_arcs[axis_index]

        world_axis = self.world_rf.xyz_lines[axis_index]
        world_axis_vec = Vector(world_axis.pos[1] - world_axis.pos[0])

        imu_axis = self.imu_rf.xyz_lines[axis_index]
        imu_origin_tred = tr.map(imu_axis.pos[0])
        imu_endpoint_tred = tr.map(imu_axis.pos[1])
        imu_axis_vec_tred = Vector(imu_endpoint_tred - imu_origin_tred)

        arc.update(world_axis_vec, imu_axis_vec_tred)

    def _update_accel_arrow(self, arrow: Cone, accel: float):
        # Min accel threshold
        if abs(accel) < 0.05:
            arrow.setVisible(False)
            return

        accel2len = 0.07
        accel2width = 0.05

        desired_length = accel * accel2len
        desired_width = accel * accel2width
        length_scale_factor = desired_length / arrow.length
        width_scale_factor = desired_width / arrow.radius[0]
        width_scale_factor = np.clip(width_scale_factor, 0.5, 1)

        tr_base = Transform3D(arrow.base_transform)
        tr_scale = Transform3D()
        tr_scale.scale(width_scale_factor, width_scale_factor, length_scale_factor)
        tr = tr_base * tr_scale

        arrow.setTransform(tr)
        arrow.setVisible(True)

    def _update_gyro_arc(self, axis_index, angular_speed):
        arc: ArcItem = self.gyro_arcs[axis_index]
        arrow: Cone = self.gyro_arrows[axis_index]
        
        axis_vec = Vector(self.world_rf.xyz_lines[axis_index].pos[1])
        axis_unit_vec = axis_vec.normalized()

        # Arc
        angular_speed_max = 7
        ang_speed2angle = 360 / angular_speed_max * angular_speed

        arc_start = arc.base_transform.map(Vector(0, 0, 0))
        tr_end = Transform3D()
        tr_end.rotate(ang_speed2angle, axis_unit_vec)
        arc_end = tr_end.map(arc_start)

        arc.update(arc_start, arc_end)
        tr = Transform3D()
        tr.translate(axis_vec)
        arc.setTransform(tr)

        # Arrow
        angular_speed2len = 0.1
        desired_length = angular_speed * angular_speed2len
        scale_factor = desired_length / arrow.length

        tr_scale = Transform3D()
        arc_translation = Vector(arc.line.pos[-1])
        arc_translation = tr.map(arc_translation)
        tr_scale.translate(arc_translation)
        tr_scale.rotate(np.rad2deg(arc.arc_angle)*np.sign(angular_speed), *axis_unit_vec.toTuple())
        if axis_index == 0:
            tr_scale.scale(1, scale_factor, 1)
        if axis_index == 1:
            tr_scale.scale(scale_factor, 1, 1)
        else:
            tr_scale.scale(1, -scale_factor, 1)

        tr = tr_scale * arrow.base_transform
        arrow.setTransform(tr)

    def _add_grids(self):
        grid_1m = GLGridItem()
        grid_1m.setSize(x=10, y=10, z=10)
        grid_1m.setSpacing(x=1, y=1, z=1)
        grid_1m.setColor((255, 255, 255, 30))
        self.addItem(grid_1m)



from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QSlider,
    QHBoxLayout,
    QLabel,
)


class IMU3DDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU 3D Demo")
        self.resize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.imu_widget = IMU3D()
        main_layout.addWidget(self.imu_widget, stretch=1)

        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)

        for axis, label in zip(["X", "Y", "Z"], ["Roll", "Pitch", "Yaw"]):
            slider_layout = QVBoxLayout()
            control_layout.addLayout(slider_layout)

            slider_label = QLabel(f"{label} (°)")
            slider_layout.addWidget(slider_label)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(-360, 360)
            slider.setValue(0)
            slider.setObjectName(f"slider_{axis.lower()}")
            slider.valueChanged.connect(self.update_orientation)
            slider_layout.addWidget(slider)
            setattr(self, f"slider_{axis.lower()}", slider)

        self.animation_angle = 0
        self.timer = QTimer()
        self.timer.start(30)

        self.prev_roll = 0.0
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.dt = self.timer.interval() / 1000.0

    def update_orientation(self):
        roll_deg = self.slider_x.value()
        pitch_deg = self.slider_y.value()
        yaw_deg = self.slider_z.value()
        
        roll = np.radians(roll_deg)
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)

        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(
            pitch / 2
        ) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(
            pitch / 2
        ) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(
            pitch / 2
        ) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(
            pitch / 2
        ) * np.sin(yaw / 2)
        quaternion = QQuaternion(qw, qx, qy, qz)

        gravity_world = Vector(0, 0, -9.81)
        accel_vector = quaternion.conjugated().rotatedVector(gravity_world)
        noise = np.random.normal(0, 0.1, 3)
        accel = Vector(
            accel_vector.x() + noise[0], accel_vector.y() + noise[1], accel_vector.z() + noise[2]
        )

        roll_rate = (roll - self.prev_roll) / self.dt
        pitch_rate = (pitch - self.prev_pitch) / self.dt
        yaw_rate = (yaw - self.prev_yaw) / self.dt
        
        self.prev_roll = roll
        self.prev_pitch = pitch
        self.prev_yaw = yaw
        
        gyro = Vector(roll_rate, pitch_rate, yaw_rate)

        self.imu_widget.update_data(quaternion, accel, gyro)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = IMU3DDemo()
    demo.show()
    sys.exit(app.exec())
