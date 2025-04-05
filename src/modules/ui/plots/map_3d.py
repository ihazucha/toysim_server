import sys

sys.path.append("C:/Users/ihazu/Desktop/projects/toysim_server/src")
import numpy as np

from copy import deepcopy

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QVector3D, QFont, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
    QPushButton,
    QGridLayout,
    QGroupBox,
)

# Set mesh data to render as solid cube
from pyqtgraph.opengl.MeshData import MeshData
from pyqtgraph.opengl.items.GLMeshItem import GLMeshItem

from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLGridItem, GLTextItem, GLBoxItem
from pyqtgraph import Transform3D

from modules.ui.plots import Colors
from datalink.data import Rotation


class MColors:
    WHITE = QColor(255, 255, 255, 255)
    GRAY = (125, 125, 125, 255)
    RED = QColor(255, 0, 0, 255)
    GREEN = QColor(0, 255, 0, 255)
    BLUE = QColor(0, 0, 255, 255)
    RED_TRANS = QColor(175, 0, 0, 200)


class Car3D:
    """Simple 3D car model for the Map3D visualization."""

    def __init__(self, parent_widget: GLViewWidget):
        """Create a simple car model starting at origin."""
        self.parent_widget = parent_widget

        chassis_len = 0.255
        chassis_width = 0.155
        chassis_height = 0.02
        wheelbase = 0.185
        track = 0.155
        wheel_radius = 0.06
        wheel_width = 0.02

        # Create car body (main chassis)
        self.body = GLBoxItem(size=QVector3D(chassis_len, chassis_width, chassis_height / 1.5))
        self.body.setColor(MColors.RED_TRANS)
        # Store initial offsets instead of translating directly
        self.body_offset = QVector3D(0, -chassis_width / 2, wheel_radius / 2)

        # Create wheels (4 black cylinders)
        self.wheels = []
        self.wheel_offsets = []

        # Wheel positions relative to car center
        wheel_positions = [
            (wheelbase, chassis_width / 2, -wheel_radius / 2),  # Front Left
            (
                wheelbase,
                -chassis_width / 2 - wheel_width,
                -wheel_radius / 2,
            ),  # Front Right
            (-wheel_radius / 2, chassis_width / 2, -wheel_radius / 2),  # Rear Left
            (-wheel_radius / 2, -chassis_width / 2 - wheel_width, -wheel_radius / 2),  # Rear Right
        ]

        for wx, wy, wz in wheel_positions:
            wheel = GLBoxItem(size=QVector3D(wheel_radius, wheel_width, wheel_radius))
            wheel.setColor(MColors.GRAY)
            # Store wheel offset instead of translating directly
            self.wheel_offsets.append(QVector3D(wx, wy, wz + wheel_radius / 2))
            self.wheels.append(wheel)

        # Add all parts to the widget
        self.add_to_widget(parent_widget)

        # Initialize position and orientation
        self.x = 0
        self.y = 0
        self.heading = 0
        self.update_position(0, 0, 0)

    def add_to_widget(self, widget):
        """Add all car components to the widget."""
        widget.addItem(self.body)
        for wheel in self.wheels:
            widget.addItem(wheel)

    def update_position(self, x, y, heading_deg):
        """Update the car's position and orientation."""
        # Store current state
        self.x = x
        self.y = y
        self.heading = heading_deg

        # Create transform for body
        transform_body = Transform3D()
        transform_body.translate(x, y, 0)
        transform_body.rotate(heading_deg, 0, 0, 1)  # Rotate around Z axis
        transform_body.translate(self.body_offset)  # Apply body offset
        self.body.setTransform(transform_body)

        # Apply transform to wheels
        for i, wheel in enumerate(self.wheels):
            transform_wheel = Transform3D()
            transform_wheel.translate(x, y, 0)
            transform_wheel.rotate(heading_deg, 0, 0, 1)
            transform_wheel.translate(self.wheel_offsets[i])  # Apply wheel offset
            wheel.setTransform(transform_wheel)


class CubeMesh(MeshData):
    VERTEXES = np.array(
        [
            [1, 1, 1],  # top front right
            [1, 1, 0],  # bottom front right
            [1, 0, 1],  # top front left
            [1, 0, 0],  # bottom front left
            [0, 1, 1],  # top back right
            [0, 1, 0],  # bottom back right
            [0, 0, 1],  # top back left
            [0, 0, 0],  # bottom back left
        ]
    )
    FACES = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],  # front
            [4, 6, 5],
            [5, 6, 7],  # back
            [0, 2, 4],
            [2, 6, 4],  # top
            [1, 5, 3],
            [3, 5, 7],  # bottom
            [0, 4, 1],
            [1, 4, 5],  # right
            [2, 3, 6],
            [3, 7, 6],  # left
        ]
    )

    def __init__(self, size: float = 1, *args, **kwargs):
        super().__init__(vertexes=self.VERTEXES * size, faces=self.FACES, *args, **kwargs)


class OpaqueCube(GLMeshItem):
    def __init__(self, size: float, color=MColors.WHITE, *args, **kwargs):
        self.size = size
        super().__init__(
            meshdata=CubeMesh(size=size),
            color=color,
            shader="shaded",
            smooth=False,
            glOptions="opaque",
            *args,
            **kwargs,
        )


class BasisVectors3D:
    BASE_FONT_SIZE = 16
    MIN_FONT_SIZE = 6
    FONT = QFont("Monospaced", BASE_FONT_SIZE)

    def __init__(self, parent_widget: GLViewWidget, name: str):
        self.parent_widget = parent_widget
        self.name = name

        size = 0.05
        self.x = GLLinePlotItem(pos=np.array([[0] * 3, [size, 0, 0]]), color=MColors.RED, width=3)
        self.y = GLLinePlotItem(pos=np.array([[0] * 3, [0, size, 0]]), color=MColors.GREEN, width=3)
        self.z = GLLinePlotItem(pos=np.array([[0] * 3, [0, 0, size]]), color=MColors.BLUE, width=3)
        self.xyz_lines = [self.x, self.y, self.z]

        # dist = size + size / 5
        # self.x_label = GLTextItem(pos=np.array([dist, size / 12, 0]), text=f"X", color=MColors.RED)
        # self.y_label = GLTextItem(pos=np.array([size / 12, dist, 0]), text=f"Y", color=MColors.GREEN)
        # self.z_label = GLTextItem(pos=np.array([0, size / 12, dist]), text=f"Z", color=MColors.BLUE)
        # self.xyz_labels = [self.x_label, self.y_label, self.z_label]

        self.origin_dot = OpaqueCube(size=0.008)
        self.origin_dot.translate(*[-0.004] * 3)

        self.name_label = GLTextItem(pos=np.array([0, 0, -0.02]), text=name, color=MColors.WHITE)

        self._add_to_parent_widget(parent_widget)

    def _add_to_parent_widget(self, widget: GLViewWidget):
        for line in self.xyz_lines:
            widget.addItem(line)
        # for label in self.xyz_labels:
        #     widget.addItem(label)

        widget.addItem(self.origin_dot)
        widget.addItem(self.name_label)

    def scale_font_by_camera_distance(self, distance: float):
        scale = 2.5
        size = self.BASE_FONT_SIZE / (distance * scale)
        size = max(self.MIN_FONT_SIZE, size)
        self.FONT.setPointSizeF(size)

        # for label in self.xyz_labels:
        #     label.setData(font=self.FONT)

        self.name_label.setData(font=self.FONT)

    def transform(self, tr: Transform3D):
        for line in self.xyz_lines:
            line.setTransform(tr)
        # for label in self.xyz_labels:
        #     label.setTransform(tr)

        self.origin_dot.setTransform(tr)
        self.name_label.setTransform(tr)


class ViewportAxes(QWidget):
    """Fixed coordinate system display for the bottom left corner that rotates with camera."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 100)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.rotation = Transform3D()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Center
        start_x = self.width() / 2
        start_y = self.height() / 2

        axis_length = 25

        # Draw axis lines - transform the direction vectors through our rotation matrix
        for axis, color, label in [
            # (X, Y, Z) World frame == (-Z, X, Y) in Image frame
            (QVector3D(0, 0, -1), QColor(255, 0, 0), "X"),
            (QVector3D(1, 0, 0), QColor(0, 255, 0), "Y"),
            (QVector3D(0, 1, 0), QColor(0, 0, 255), "Z"),
        ]:
            # Draw in 2D
            transformed = self.rotation.map(axis)
            end_x = start_x + transformed.x() * axis_length
            end_y = start_y - transformed.y() * axis_length

            # Line
            pen = QPen(color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(start_x, start_y, end_x, end_y)

            # Label
            font = painter.font()
            font.setBold(True)
            font.setFamily("Consolas, Helvetica")
            painter.setFont(font)
            painter.drawText(end_x - 3.5, end_y - 5, label)

    def update_rotation(self, glview_widget):
        """Update rotation based on camera's view matrix."""
        elevation = glview_widget.opts["elevation"]
        azimuth = glview_widget.opts["azimuth"]

        self.rotation = Transform3D()
        self.rotation.rotate(azimuth, 0, 1, 0)  # Yaw
        self.rotation.rotate(elevation, 1, 0, 0)  # Pitch
        self.update()


class Map3D(GLViewWidget):
    INIT_POS = QVector3D(0, 0, 0)
    INIT_DIST = 0.2
    INIT_AZIMUTH = 135

    def __init__(self):
        super().__init__()
        self.setBackgroundColor(Colors.FOREGROUND)
        self.setCameraPosition(
            pos=self.INIT_POS, distance=self.INIT_DIST, azimuth=self.INIT_AZIMUTH
        )
        self._add_grid()

        self._top_down_view_enabled = False
        self._stored_camera_state = None

        # React to shortcuts without focus
        self.setFocusPolicy(Qt.StrongFocus)

        # Reference frames
        self.world_basis_vectors = BasisVectors3D(parent_widget=self, name="W(0, 0, 0)")
        self.world_basis_vectors.scale_font_by_camera_distance(distance=Map3D.INIT_DIST)

        self.viewport_axes = ViewportAxes(self)
        self.viewport_axes.update_rotation(self)
        self.viewport_axes.show()

        self.car = Car3D(parent_widget=self)

        # Top-down coordinate label
        self.coordinate_label = QLabel(self)
        self.coordinate_label.setStyleSheet(
            """
            background-color: rgba(0, 0, 0, 120); 
            color: white; 
            padding: 5px;
            border-radius: 3px;
        """
        )
        self.coordinate_label.setAlignment(Qt.AlignCenter)
        self.coordinate_label.setMinimumWidth(150)
        self.coordinate_label.hide()

    def move_car(self, x, y, heading=None):
        if hasattr(self, "car"):
            if heading is None:
                heading = self.car.heading
            self.car.update_position(x, y, heading)

    def _add_grid(self):
        grid_1m = GLGridItem()
        grid_1m.setSize(x=10, y=10, z=10)
        grid_1m.setSpacing(x=1, y=1, z=1)
        self.addItem(grid_1m)

        grid_10cm = GLGridItem()
        grid_10cm.setSize(x=10, y=10, z=1)
        grid_10cm.setSpacing(x=0.1, y=0.1, z=0.1)
        grid_10cm.setColor((255, 255, 255, 25))
        self.addItem(grid_10cm)

    def toggle_top_down_view(self):
        """Toggle between top-down view and normal 3D view."""
        self._top_down_view_enabled = not self._top_down_view_enabled

        if self._top_down_view_enabled:
            self._store_camera_state()
            self.setCameraPosition(elevation=90, azimuth=180)
            self.setMouseTracking(True)
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.coordinate_label.show()
        else:
            self._restore_camera_state()
            self.setMouseTracking(False)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.coordinate_label.hide()

    def _store_camera_state(self):
        self._stored_camera_state = deepcopy(self.opts)

    def _restore_camera_state(self):
        if self._stored_camera_state:
            self.setCameraPosition(
                distance=self._stored_camera_state["distance"],
                elevation=self._stored_camera_state["elevation"],
                azimuth=self._stored_camera_state["azimuth"],
            )
            if "center" in self._stored_camera_state and hasattr(self, "opts"):
                self.opts["center"] = self._stored_camera_state["center"]

    def mouseMoveEvent(self, ev):
        self.viewport_axes.update_rotation(self)

        # For top-down view only panning and zoom are enabled
        if self._top_down_view_enabled:
            self._update_mouse_position_display(ev)

            if ev.buttons() == Qt.LeftButton:
                delta = ev.position().toPoint() - self._last_mouse_position.toPoint()
                self._last_mouse_position = ev.position()
                pan_speed = 0.02
                dx = pan_speed * delta.x()
                dy = pan_speed * delta.y()
                self.pan(dy, dx, 0)
        else:
            return super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev) -> None:
        super().mousePressEvent(ev)
        # Initialize for top-down view mouseMoveEvent
        self._last_mouse_position = ev.position()

    def wheelEvent(self, ev):
        """Override wheel event to update text scaling after zoom."""
        super().wheelEvent(ev)
        self.viewport_axes.update_rotation(self)
        if self._top_down_view_enabled and ev.position() is not None:
            self._update_mouse_position_display(ev)
        if hasattr(self, "world_basis_vectors"):
            self.world_basis_vectors.scale_font_by_camera_distance(self.opts["distance"])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_T:
            self.toggle_top_down_view()
            event.accept()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        """Handle resize to reposition viewport axes in bottom left corner."""
        super().resizeEvent(event)
        if hasattr(self, "viewport_axes"):
            self.viewport_axes.move(10, self.height() - self.viewport_axes.height() - 10)

    def _update_mouse_position_display(self, ev):
        pos = ev.position().toPoint()
        world_x, world_y = self.top_down_screen_to_world(pos.x(), pos.y())

        self.coordinate_label.setText(f"X: {world_x:.3f}, Y: {world_y:.3f}")

        label_x = pos.x() + 15
        label_y = pos.y() - 30

        # Stay within widget bounds
        if label_x + self.coordinate_label.width() > self.width():
            label_x = pos.x() - self.coordinate_label.width() - 5
        if label_y < 0:
            label_y = pos.y() + 15

        self.coordinate_label.move(label_x, label_y)

    def top_down_screen_to_world(self, screen_x, screen_y):
        viewport_width = self.width()
        viewport_height = self.height()
        aspect_ratio = viewport_width / viewport_height

        camera_distance = self.opts["distance"]
        camera_center = self.opts["center"]
        fov = self.opts.get("fov", 60)

        # Visible ground plane area
        visible_height = 2.0 * camera_distance * np.tan(np.deg2rad(fov / 2))
        visible_width = visible_height

        # <-1, 1> Normalized
        norm_x = (2.0 * screen_x / viewport_width) - 1.0
        norm_y = 1.0 - (2.0 * screen_y / viewport_height)

        # World (X, Y) are Viewport (Y, -X)
        world_x = (camera_center.x() * aspect_ratio) + (norm_y * visible_height / 2.0)
        world_y = (camera_center.y()) - (norm_x * visible_width / 2.0)

        # Correct for aspect ratio
        return world_x / aspect_ratio, world_y

    def update_data(self, rotation: Rotation):
        roll, pitch, yaw = (
            np.deg2rad(rotation.roll),
            np.deg2rad(rotation.pitch),
            np.deg2rad(rotation.yaw),
        )

        transform = Transform3D()
        transform.rotate(np.rad2deg(roll), 1, 0, 0)
        transform.rotate(np.rad2deg(pitch), 0, 1, 0)
        transform.rotate(np.rad2deg(yaw), 0, 0, 1)

        self.world_basis_vectors.transform(transform)


# Demo
# -------------------------------------------------------------------------------------------------


# Replace the existing Map3DDemo.__init__ method with this implementation


class ControlPanel(QWidget):
    """Floating control panel that can be toggled on/off."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("3D Controls")

        # Main layout
        layout = QVBoxLayout(self)

        # Rotation controls
        rotation_group = QGroupBox("Rotation")
        rotation_layout = QVBoxLayout(rotation_group)
        self.roll_slider = self._create_slider_group("Roll", rotation_layout)
        self.pitch_slider = self._create_slider_group("Pitch", rotation_layout)
        self.yaw_slider = self._create_slider_group("Yaw", rotation_layout)
        layout.addWidget(rotation_group)

        # Animation controls
        animation_group = QGroupBox("Animation")
        animation_layout = QVBoxLayout(animation_group)

        buttons_layout = QHBoxLayout()
        animation_layout.addLayout(buttons_layout)

        self.animate_button = QPushButton("Start Animation")
        self.animate_button.setCheckable(True)
        buttons_layout.addWidget(self.animate_button)

        reset_button = QPushButton("Reset")
        buttons_layout.addWidget(reset_button)

        layout.addWidget(animation_group)

        # Car controls
        car_group = QGroupBox("Car Controls")
        car_layout = QVBoxLayout(car_group)

        car_sliders_layout = QGridLayout()
        car_layout.addLayout(car_sliders_layout)

        # X position
        car_sliders_layout.addWidget(QLabel("X Position:"), 0, 0)
        self.car_x_slider = QSlider(Qt.Horizontal)
        self.car_x_slider.setMinimum(-50)
        self.car_x_slider.setMaximum(50)
        self.car_x_slider.setValue(0)
        car_sliders_layout.addWidget(self.car_x_slider, 0, 1)

        # Y position
        car_sliders_layout.addWidget(QLabel("Y Position:"), 1, 0)
        self.car_y_slider = QSlider(Qt.Horizontal)
        self.car_y_slider.setMinimum(-50)
        self.car_y_slider.setMaximum(50)
        self.car_y_slider.setValue(0)
        car_sliders_layout.addWidget(self.car_y_slider, 1, 1)

        # Heading
        car_sliders_layout.addWidget(QLabel("Heading:"), 2, 0)
        self.car_heading_slider = QSlider(Qt.Horizontal)
        self.car_heading_slider.setMinimum(0)
        self.car_heading_slider.setMaximum(359)
        self.car_heading_slider.setValue(0)
        car_sliders_layout.addWidget(self.car_heading_slider, 2, 1)

        layout.addWidget(car_group)

        # Connect button signals - these will be connected in the main class
        self.reset_button = reset_button

        # Size policy
        self.setMinimumWidth(300)

    def _create_slider_group(self, name, parent_layout):
        """Create a labeled slider for a rotation axis."""
        layout = QHBoxLayout()
        label = QLabel(f"{name}: 0°")
        label.setMinimumWidth(60)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(-180)
        slider.setMaximum(180)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(45)

        # Store the associated label
        slider.valueChanged.connect(lambda val: label.setText(f"{name}: {val}°"))

        layout.addWidget(QLabel(name))
        layout.addWidget(slider)
        layout.addWidget(label)

        parent_layout.addLayout(layout)
        return slider


class Map3DDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Visualization Demo")
        self.resize(800, 600)

        # Create central widget with only the 3D map
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create 3D map widget (takes full space now)
        self.map_3d = Map3D()
        main_layout.addWidget(self.map_3d)

        # # Create floating control panel
        # self.control_panel = ControlPanel()

        # # Connect controls
        # self.control_panel.roll_slider.valueChanged.connect(self._update_orientation)
        # self.control_panel.pitch_slider.valueChanged.connect(self._update_orientation)
        # self.control_panel.yaw_slider.valueChanged.connect(self._update_orientation)

        # self.control_panel.animate_button.clicked.connect(self._toggle_animation)
        # self.control_panel.reset_button.clicked.connect(self._reset_orientation)

        # self.control_panel.car_x_slider.valueChanged.connect(self._update_car_position)
        # self.control_panel.car_y_slider.valueChanged.connect(self._update_car_position)
        # self.control_panel.car_heading_slider.valueChanged.connect(self._update_car_position)

        # # Animation timer
        # self.animation_timer = QTimer()
        # self.animation_timer.timeout.connect(self._update_animation)
        # self.animation_angle = 0

        # # Add control panel toggle action
        # toggle_controls_action = self.menuBar().addAction("Toggle Controls")
        # toggle_controls_action.triggered.connect(self._toggle_control_panel)
        # toggle_controls_action.setShortcut("Ctrl+C")

        # Initial update
        # self._update_orientation()

    def _toggle_control_panel(self):
        if self.control_panel.isVisible():
            self.control_panel.hide()
        else:
            self.control_panel.show()

    def closeEvent(self, event):
        # Also close the control panel when the main window is closed
        self.control_panel.close()
        super().closeEvent(event)

    # Update the methods to use control_panel's sliders
    def _update_orientation(self):
        """Update the 3D view with current rotation values."""
        rotation = Rotation(
            roll=self.control_panel.roll_slider.value(),
            pitch=self.control_panel.pitch_slider.value(),
            yaw=self.control_panel.yaw_slider.value(),
        )
        self.map_3d.update_data(rotation)

    def _update_car_position(self):
        """Update car position based on slider values."""
        x = self.control_panel.car_x_slider.value() / 10.0
        y = self.control_panel.car_y_slider.value() / 10.0
        heading = self.control_panel.car_heading_slider.value()
        self.map_3d.move_car(x, y, heading)

    def _toggle_animation(self, checked):
        """Toggle animation on/off."""
        if checked:
            self.control_panel.animate_button.setText("Stop Animation")
            self.animation_timer.start(30)
            # Disable sliders during animation
            self.control_panel.roll_slider.setEnabled(False)
            self.control_panel.pitch_slider.setEnabled(False)
            self.control_panel.yaw_slider.setEnabled(False)
        else:
            self.control_panel.animate_button.setText("Start Animation")
            self.animation_timer.stop()
            # Enable sliders after animation
            self.control_panel.roll_slider.setEnabled(True)
            self.control_panel.pitch_slider.setEnabled(True)
            self.control_panel.yaw_slider.setEnabled(True)

    def _update_animation(self):
        """Update animation frame."""
        self.animation_angle += 2
        if self.animation_angle >= 360:
            self.animation_angle = 0

        # Create a rotation that changes over time
        rotation = Rotation(
            roll=45 * np.sin(np.deg2rad(self.animation_angle)),
            pitch=45 * np.sin(np.deg2rad(self.animation_angle + 120)),
            yaw=self.animation_angle,
        )

        # Update slider positions
        self.control_panel.roll_slider.setValue(int(rotation.roll))
        self.control_panel.pitch_slider.setValue(int(rotation.pitch))
        self.control_panel.yaw_slider.setValue(int(rotation.yaw))

        # Update the 3D view
        self.map_3d.update_data(rotation)

    def _reset_orientation(self):
        """Reset to default orientation."""
        self.control_panel.roll_slider.setValue(0)
        self.control_panel.pitch_slider.setValue(0)
        self.control_panel.yaw_slider.setValue(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Map3DDemo()
    window.show()
    sys.exit(app.exec())
