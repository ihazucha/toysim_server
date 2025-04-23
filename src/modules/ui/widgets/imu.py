import sys

sys.path.append("C:/Users/ihazu/Desktop/projects/toysim_server/src")

import numpy as np
from typing import Tuple, Iterable

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

        x_line = np.array([[0] * 3, [size, 0, 0]])
        x_color = QColor(MColors.RED)
        x_color.setAlpha(alpha)
        self.x = GLLinePlotItem(pos=x_line, color=x_color, width=width, antialias=True)
        y_line = np.array([[0] * 3, [0, size, 0]])
        y_color = QColor(MColors.GREEN)
        y_color.setAlpha(alpha)
        self.y = GLLinePlotItem(pos=y_line, color=y_color, width=width, antialias=True)
        z_line = np.array([[0] * 3, [0, 0, size]])
        z_color = QColor(MColors.BLUE)
        z_color.setAlpha(alpha)
        self.z = GLLinePlotItem(pos=z_line, color=z_color, width=width, antialias=True)

        self.xyz_lines = [self.x, self.y, self.z]

        radius = size / 4
        self.origin_offset = Vector(*[-radius / 2] * 3)
        self.origin = OpaqueCube(size=radius, shader=customNormalColor, glOptions="translucent")
        self.origin.setDepthValue(-1)

        self._add_to_parent_widget(parent_widget)

        self.setTransform(Transform3D())

    def get_position(self):
        return Vector(self.x.transform().matrix()[:3, 3])

    def get_transform(self):
        return self.x.transform()

    def setTransform(self, tr: Transform3D):
        for line in self.xyz_lines:
            line.setTransform(tr)

        tr.translate(self.origin_offset)
        self.origin.setTransform(tr)

    def _add_to_parent_widget(self, widget: GLViewWidget):
        for line in self.xyz_lines:
            widget.addItem(line)

        if self.show_cube:
            widget.addItem(self.origin)


class ArcItem():
    def __init__(self, segments=32, radius=1, width=1, color: QColor=MColors.WHITE):
        self.segments=segments
        self.radius=radius

        self.line = GLLinePlotItem(color=color, width=width, antialias=True)

    def update(self, start_vec: Vector, end_vec: Vector):
        # Get unit
        start_norm = np.linalg.norm(start_vec)
        end_norm = np.linalg.norm(end_vec)
        start_unit = start_vec / start_norm
        end_unit = end_vec / end_norm

        # Angle between vectors
        dot_product = np.dot(start_unit, end_unit)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = np.arccos(dot_product)

        rotation_axis = np.cross(start_unit, end_unit)
        norm_axis = np.linalg.norm(rotation_axis)

        # Skip small angles
        MIN_ANGLE_RAD = 0.005  # ~0.3 degrees
        # Skip parallel
        MIN_AXIS_NORM = 1e-6
        if norm_axis <= MIN_AXIS_NORM or angle_rad <= MIN_ANGLE_RAD:
            self.line.setData(pos=np.empty((0, 3)))
            self.line.setVisible(False)
            return

        rotation_axis /= norm_axis
        arc_points = []

        for i in range(self.line + 1):
            arc_step = i / (self.line)
            angle_step = angle_rad * arc_step

            tr = Transform3D()
            tr.rotate(np.degrees(angle_step), *rotation_axis)
            # Rotate origin vector to the position on the arc
            rotated_vector = np.array(tr.map(Vector(*start_vec)))
            # Radius as factor of the original axis length
            rotated_vector *= self.radius

            arc_points.append(rotated_vector)

        self.line.setData(pos=np.array(arc_points))
        self.line.setVisible(True)


class IMU3D(GLViewWidget):
    INIT_OPTS = {
        "center": Vector(0, 0, 0),
        "distance": 2.0,
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
        self._add_grids()
        self.world_rf = IMUReferenceFrame(parent_widget=self, size=1, alpha=128, show_cube=False)
        self.imu_rf = IMUReferenceFrame(parent_widget=self, size=0.5, width=5)
        self._add_angle_arcs()
        self._add_accel_arrows()


    def _add_angle_arcs(self):
        self.angle_arcs = {}
        for axis, color in (
            ("x", QColor(MColors.RED)),
            ("y", QColor(MColors.GREEN)),
            ("z", QColor(MColors.BLUE)),
        ):
            color.setAlpha(80)
            arc_item = GLLinePlotItem(color=color, width=2, antialias=True)
            self.addItem(arc_item)
            self.angle_arcs[axis] = arc_item

    def _add_accel_arrows(self):
        self.accel_arrows = {}
        for axis, color, offset in (
            ("x", QColor(MColors.RED), (90, 0, 1, 0)),
            ("y", QColor(MColors.GREEN), (-90, 1, 0, 0)),
            ("z", QColor(MColors.BLUE), (0, 0, 0, 1))
        ):
            color.setAlpha(80)
            world_rf_axis = getattr(self.world_rf, axis)
            
            tr = Transform3D()
            tr.translate(*world_rf_axis.pos[1])
            tr.rotate(*offset)
            
            arrow = Cone(color=color, length=0.1, radius=(0.02, 0), base_transform=tr)

            self.accel_arrows[axis] = arrow
            self.addItem(arrow)
        

    def update_data(self, quaternion: QQuaternion, accel: Vector, gyro: Vector):
        # Apply transformation to IMU reference frame
        transform = Transform3D()
        transform.rotate(quaternion)
        self.imu_rf.setTransform(transform)

        roll, pitch, yaw = quat2euler(quaternion)
        roll_deg, pitch_deg, yaw_deg = np.rad2deg([roll, pitch, yaw])
        print(f"\rRoll: {roll_deg:.2f}°, Pitch: {pitch_deg:.2f}°, Yaw: {yaw_deg:.2f}°", end="")

        for k in self.angle_arcs.keys():
            self._update_angle_arc(k, transform)

        for k in self.accel_arrows.keys():
            if k == "x":
                axis_accel = accel[0]
            elif k =="y":
                axis_accel = accel[1]
            else:
                axis_accel = accel[2]
                
            self._update_accel_arrow(k, axis_accel, transform)

    def _update_accel_arrow(self, axis_name, axis_accel: float, transform: Transform3D):
        arrow: Cone = self.accel_arrows[axis_name]

        MIN_ACCEL_DISPLAY_THRESHOLD = 0.05
        if abs(axis_accel) < MIN_ACCEL_DISPLAY_THRESHOLD:
            arrow.setVisible(False)
            return

        BASE_CONE_LENGTH = 0.1
        ACCEL_TO_LENGTH_FACTOR = 0.02
        desired_length = axis_accel * ACCEL_TO_LENGTH_FACTOR
        scale_factor = desired_length / BASE_CONE_LENGTH

        tr = Transform3D(arrow.base_transform)
        tr2 = Transform3D()
        tr2.scale(1, 1, scale_factor)
        tr_final = tr * tr2

        arrow.setTransform(tr_final)
        arrow.setVisible(True)
        

    def _update_angle_arc(self, axis_name: str, transform: Transform3D, arc_points_num=30, radius=0.5):
        angle_arc_line = self.angle_arcs[axis_name]

        # World axis vector
        world_axis = getattr(self.world_rf, axis_name)
        world_origin = world_axis.pos[0]
        world_endpoint = world_axis.pos[1]
        world_axis_vec = world_endpoint - world_origin

        # IMU axis vector
        imu_axis = getattr(self.imu_rf, axis_name)
        imu_origin = np.array(transform.map(Vector(*imu_axis.pos[0])))
        imu_endpoint = np.array(transform.map(Vector(*imu_axis.pos[1])))
        imu_axis_vec = imu_endpoint - imu_origin

        # Normalized
        norm_world = np.linalg.norm(world_axis_vec)
        unit_world = world_axis_vec / norm_world
        norm_imu = np.linalg.norm(imu_axis_vec)
        unit_imu = imu_axis_vec / norm_imu

        # Angle between vectors
        dot_product = np.dot(unit_world, unit_imu)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = np.arccos(dot_product)

        rotation_axis = np.cross(unit_world, unit_imu)
        norm_axis = np.linalg.norm(rotation_axis)

        # Skip small angles
        MIN_ANGLE_RAD = 0.005  # ~0.3 degrees
        # Skip parallel
        MIN_AXIS_NORM = 1e-6
        if norm_axis <= MIN_AXIS_NORM or angle_rad <= MIN_ANGLE_RAD:
            angle_arc_line.setData(pos=np.empty((0, 3)))
            angle_arc_line.setVisible(False)
            return

        rotation_axis /= norm_axis
        arc_points = []

        for i in range(arc_points_num + 1):
            arc_step = i / (arc_points_num)
            angle_step = angle_rad * arc_step

            tr = Transform3D()
            tr.rotate(np.degrees(angle_step), *rotation_axis)
            # Rotate origin vector to the position on the arc
            rotated_vector = np.array(tr.map(Vector(*world_axis_vec)))
            # Radius as factor of the original axis length
            rotated_vector *= radius

            final_point = world_origin + rotated_vector
            arc_points.append(final_point)

        if len(arc_points) > 1:
            angle_arc_line.setData(pos=np.array(arc_points))
            angle_arc_line.setVisible(True)

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

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # IMU widget
        self.imu_widget = IMU3D()
        main_layout.addWidget(self.imu_widget, stretch=1)

        # Control panel
        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)

        # Rotation sliders
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

        # Timer for animation and data generation
        self.animation_angle = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_orientation) # Update orientation periodically
        # self.timer.timeout.connect(self.animate_rotation) # Optionally connect animation
        self.timer.start(30)  # Update roughly 33 times per second

        # Store previous orientation for gyro calculation
        self.prev_roll = 0.0
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.dt = self.timer.interval() / 1000.0 # Time step in seconds

    def update_orientation(self):
        # --- Get current rotation angles ---
        roll_deg = self.slider_x.value()
        pitch_deg = self.slider_y.value()
        yaw_deg = self.slider_z.value()
        roll = np.radians(roll_deg)
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)

        # --- Convert Euler angles to quaternion ---
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

        # --- Simulate Accelerometer Data (Gravity Vector in IMU Frame) ---
        # Assume gravity points down along the world Z-axis
        gravity_world = Vector(0, 0, -9.81)
        # Rotate the world gravity vector into the IMU's frame using the inverse rotation
        # QQuaternion.rotatedVector is equivalent to q * v * q^-1
        accel_vector = quaternion.conjugated().rotatedVector(gravity_world)
        # Add some noise (optional)
        noise = np.random.normal(0, 0.1, 3)
        accel = Vector(accel_vector.x() + noise[0], accel_vector.y() + noise[1], accel_vector.z() + noise[2])


        # --- Simulate Gyroscope Data (Angular Velocity) ---
        # Approximate angular velocity from change in Euler angles
        # Note: This is a simplification and not perfectly accurate, especially near gimbal lock
        roll_rate = (roll - self.prev_roll) / self.dt
        pitch_rate = (pitch - self.prev_pitch) / self.dt
        yaw_rate = (yaw - self.prev_yaw) / self.dt
        # Store current angles for next calculation
        self.prev_roll = roll
        self.prev_pitch = pitch
        self.prev_yaw = yaw
        # Combine into a vector (assuming rates are in radians/sec)
        gyro = Vector(roll_rate, pitch_rate, yaw_rate)


        # --- Update IMU Widget ---
        self.imu_widget.update_data(quaternion, accel, gyro)

    def animate_rotation(self):
        # Increase the animation angle
        self.animation_angle -= 2
        # Keep angle within a reasonable range if needed, or let sliders handle it
        # if self.animation_angle <= -360:
        #     self.animation_angle += 720
        # if self.animation_angle >= 360:
        #     self.animation_angle -= 720

        # Set the slider values to create a rotation animation
        # Using setValue will trigger update_orientation via the signal
        self.slider_x.setValue(self.animation_angle % 360) # Use modulo for continuous feel
        self.slider_y.setValue((self.animation_angle + 120) % 360)
        self.slider_z.setValue((self.animation_angle + 240) % 360)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = IMU3DDemo()
    # Connect animation function to timer *after* initialization
    demo.timer.timeout.connect(demo.animate_rotation)
    demo.show()
    sys.exit(app.exec())