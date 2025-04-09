import sys

sys.path.append("C:/Users/ihazu/Desktop/projects/toysim_server/src")
import numpy as np
from typing import Any, Iterable, Set
from copy import deepcopy

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QVector4D
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

from pyqtgraph.opengl import GLViewWidget, GLGridItem, GLScatterPlotItem, GLLinePlotItem
from pyqtgraph import Transform3D, Vector

from modules.ui.plots import Colors
from modules.ui.presets import MColors

from modules.ui.widgets.opengl.shapes import OpaqueCylinder
from modules.ui.widgets.opengl.models import Car3D
from modules.ui.widgets.opengl.helpers import BasisVectors3D


class PositionLabel2D(QLabel):
    """2D floating coordinate label (e.g. for top-down view)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet(
            """
            background-color: rgba(0, 0, 0, 120); 
            color: white; 
            padding: 5px;
            border-radius: 3px;
        """
        )
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumWidth(150)
        self.hide()

    def update_coordinates(self, x: float, y: float):
        self.setText(f"X: {x:.3f}, Y: {y:.3f}")

    def toggle(self, show: bool):
        self.show() if show else self.hide()

    def update_position(
        self, x: int, y: int, x_offset: int = 15, y_offset: int = -30, check_parent_bounds=True
    ):
        # Show a lil to the side
        x += x_offset
        y += y_offset

        # Stay within widget bounds
        if check_parent_bounds:
            if x + self.width() > self.parent().width():
                x = x - self.width() - 5
            if y < 0:
                y = y + 15

        self.move(x, y)


class ViewportAxes(QWidget):
    """2D Basis Vectors display"""

    def __init__(self, parent, axis_length=25):
        self.axis_length = axis_length
        super().__init__(parent)
        self.setFixedSize(100, 100)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.show()

        self.rotation = Transform3D()

    def on_camera_change(self, opts: dict):
        self.update_rotation(elevation=opts["elevation"], azimuth=opts["azimuth"])

    def update_rotation(self, elevation: float, azimuth: float):
        self.rotation = Transform3D()
        self.rotation.rotate(-elevation, 1, 0, 0)  # Pitch
        self.rotation.rotate(azimuth, 0, 1, 0)  # Yaw
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = painter.font()
        font.setBold(True)
        font.setFamily("Monospace")
        pen = QPen()
        pen.setWidth(2)
        painter.setFont(font)

        start_x = self.width() / 2
        start_y = self.height() / 2

        # (X, Y, Z) world frame == (-Z, X, Y) image frame
        axes = [
            ("X", Vector(0, 0, -1), MColors.RED),
            ("Y", Vector(1, 0, 0), MColors.GREEN),
            ("Z", Vector(0, 1, 0), MColors.BLUE),
        ]

        for label, rotation, color in axes:
            transformed = self.rotation.map(rotation)
            end_x = start_x + transformed.x() * self.axis_length
            end_y = start_y - transformed.y() * self.axis_length

            # Line
            pen.setColor(color)
            painter.setPen(pen)
            painter.drawLine(start_x, start_y, end_x, end_y)

            # Label
            painter.drawText(end_x - 3.5, end_y - 5, label)


class ViewTransition:
    def __init__(self, animation_callback: callable, total_frames=30):
        """opts hold the GLView projection (camera) configuration"""
        self.animation_callback = animation_callback
        self.total_frames = total_frames
        self.current_frame = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self._animate_view_transition)

        self.origin_opts: dict = None
        self.target_opts: dict = None

    def set_origin(self, opts: dict):
        self.origin_opts = deepcopy(opts)

    def set_target(
        self,
        opts: dict,
        center: Vector = None,
        distance: float = None,
        elevation: float = None,
        azimuth: float = None,
    ):
        self.target_opts = deepcopy(opts)
        if center:
            self.target_opts["center"] = center
        if distance:
            self.target_opts["distance"] = distance
        if elevation:
            self.target_opts["elevation"] = elevation
        if azimuth:
            self.target_opts["azimuth"] = azimuth

    def is_transitioning(self) -> bool:
        return self.timer.isActive()

    def stop(self):
        self.timer.stop()

    def start(self, dt_ms: int):
        self.current_frame = 0
        self.timer.start(dt_ms)

    def _animate_view_transition(self):
        """Update camera position for a smooth transition between views."""
        if self.current_frame >= self.total_frames:
            self.stop()
            return

        # Calculate progress and apply easing
        progress = self.current_frame / self.total_frames
        eased_progress = 0.5 - 0.5 * np.cos(progress * np.pi)

        # Interpolate
        new_distance = self._lerp_opt("distance", eased_progress)
        new_elevation = self._lerp_opt("elevation", eased_progress)
        new_azimuth = self._lerp_opt("azimuth", eased_progress)
        new_center = self._lerp_opt("center", eased_progress)

        self.animation_callback(
            center=new_center, distance=new_distance, elevation=new_elevation, azimuth=new_azimuth
        )
        self.current_frame += 1

    def _lerp_opt(self, opt: str, t: float):
        return self._lerp(self.origin_opts[opt], self.target_opts[opt], t)

    def _lerp(self, start, end, t):
        """Linear interpolation"""
        return start + t * (end - start)


class ViewMovement:
    MOVE_SPEED = 0.02
    SPRINT_MULTIPLIER = 3.0

    def __init__(self, apply_movement_callback: callable):
        self._keys_pressed = set()
        self._timer = QTimer()
        self._timer.timeout.connect(apply_movement_callback)
        self._timer.start(16)

    def is_key_pressed(self, key) -> bool:
        return key in self._keys_pressed

    def is_movement_key(self, key) -> bool:
        return key in [
            Qt.Key.Key_W,
            Qt.Key.Key_A,
            Qt.Key.Key_S,
            Qt.Key.Key_D,
            Qt.Key.Key_Q,
            Qt.Key.Key_E,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Control,
        ]

    def set_pressed_key(self, key):
        self._keys_pressed.add(key)

    def unset_pressed_key(self, key):
        if key in self._keys_pressed:
            self._keys_pressed.remove(key)

    def get_speed(self, elevation, azimuth, topdown_view=False) -> Vector:
        # Speed + modifiers
        speed_factor = self.MOVE_SPEED
        if self.is_key_pressed(Qt.Key.Key_Shift):
            speed_factor *= self.SPRINT_MULTIPLIER
        if self.is_key_pressed(Qt.Key.Key_Control):
            speed_factor /= self.SPRINT_MULTIPLIER

        # Directions
        forward = self.get_camera_forward_vector(elevation, azimuth)
        if topdown_view:
            # Make W/S pan up/down
            forward.setX(-forward.z())
            forward.setZ(0)
        right = self.get_camera_right_vector(azimuth)
        up = Vector(0, 0, 1)

        speed = Vector(0, 0, 0)

        if self.is_key_pressed(Qt.Key.Key_W):
            speed += forward
        if self.is_key_pressed(Qt.Key.Key_S):
            speed -= forward
        if self.is_key_pressed(Qt.Key.Key_A):
            speed -= right
        if self.is_key_pressed(Qt.Key.Key_D):
            speed += right
        if self.is_key_pressed(Qt.Key.Key_Q):
            speed -= up
        if self.is_key_pressed(Qt.Key.Key_E):
            speed += up

        magnitude = speed.length()
        if magnitude > 0:
            speed /= magnitude
            speed *= speed_factor
            return speed

        return Vector(0, 0, 0)

    def get_camera_forward_vector(self, elevation, azimuth):
        elevation_rad = np.radians(elevation)
        azimuth_rad = np.radians(azimuth)
        x = -np.cos(azimuth_rad) * np.cos(elevation_rad)
        y = -np.sin(azimuth_rad) * np.cos(elevation_rad)
        z = -np.sin(elevation_rad)
        return Vector(x, y, z)

    def get_camera_right_vector(self, azimuth):
        azimuth_rad = np.radians(azimuth)
        x = -np.sin(azimuth_rad)
        y = np.cos(azimuth_rad)
        z = 0
        return Vector(x, y, z)


class NavigationData:
    def __init__(self, parent_widget: GLViewWidget):
        self.parent_widget = parent_widget
        self.roadmarks = []

        self.roadmarks_scatter = GLScatterPlotItem(
            size=10, color=(23 / 255, 155 / 255, 93 / 255, 1)
        )
        self.roadmarks_scatter.setDepthValue(1)
        self.parent_widget.addItem(self.roadmarks_scatter)
        self.path = GLLinePlotItem(
            mode="line_strip", color=(23 / 255, 155 / 255, 93 / 255, 1), width=3, antialias=True
        )
        self.parent_widget.addItem(self.path)

        self._timer = QTimer()
        self._timer.timeout.connect(self.play_next_frame)
        self._timer.start(33)

        from utils.paths import record_path
        from modules.recorder import RecordReader
        from datalink.data import ProcessedRealData

        path = record_path("1744064791554416900")
        self.data: list[ProcessedRealData] = RecordReader.read_all(path, ProcessedRealData)
        self.idx = 0

    def play_next_frame(self):
        rm_data = self.data[self.idx].roadmarks_data
        self.update(rm_data.roadmarks, rm_data.path)
        self.idx = (self.idx + 1) % len(self.data)

    def update(self, roadmarks: np.ndarray, path: np.ndarray):
        self._clear()
        for rm in roadmarks:
            mesh = OpaqueCylinder(
                radius=0.02, height=0.001, color=(0.8, 0, 0, 0.2), drawEdges=False, drawFaces=True
            )
            mesh.setDepthValue(-1)
            mesh.translate(dx=rm[0], dy=rm[1], dz=-0.001)
            self.parent_widget.addItem(mesh)
            self.roadmarks.append(mesh)
        roadmarks3d = np.column_stack((roadmarks, np.zeros(roadmarks.shape[0]) + 0.001))
        self.roadmarks_scatter.setData(pos=roadmarks3d)
        path3d = np.column_stack((path, np.zeros(path.shape[0])))
        self.path.setData(pos=path3d)

    def _clear(self):
        for rm in self.roadmarks:
            self.parent_widget.removeItem(rm)
        self.roadmarks = []


class Map3D(GLViewWidget):
    INIT_OPTS = {
        "center": Vector(-0.5, 0.5, 0.5),
        "distance": 0.1,
        "elevation": 30,
        "azimuth": 135,
    }

    MOVE_SPEED = 0.01
    SPRINT_MULTIPLIER = 3.0

    def __init__(self):
        super().__init__()
        self.setBackgroundColor((10, 10, 10, 255))
        self.setCameraPosition(
            pos=self.INIT_OPTS["center"],
            distance=self.INIT_OPTS["distance"],
            elevation=self.INIT_OPTS["elevation"],
            azimuth=self.INIT_OPTS["azimuth"],
        )
        self.setFocusPolicy(Qt.StrongFocus)  # Accept (keyboard) events without focus

        # Items
        self._add_grids()
        self.world_origin = BasisVectors3D(parent_widget=self, name="W")
        self.viewport_axes = ViewportAxes(self)
        self.car = Car3D(parent_widget=self)
        self.mouse_position_2d_label = PositionLabel2D(parent=self)
        self.navigationData = NavigationData(self)
        self._on_camera_change_group: Set[Any] = {self.viewport_axes}

        # Views
        self.view_transition = ViewTransition(animation_callback=self._animation_callback)
        self._top_down_view_enabled = False

        # Movement
        self.view_movement = ViewMovement(apply_movement_callback=self._view_movement_callback)

        self._update_on_camera_change()

    # Replace the toggle_top_down_view method with this animation version
    def toggle_top_down_view(self):
        """Toggle between top-down view and normal 3D view with animation."""
        self._top_down_view_enabled = not self._top_down_view_enabled

        # Cancel ongoing animation
        if self.view_transition.is_transitioning():
            self.view_transition.stop()

        if self._top_down_view_enabled:
            # Prevent redundant rotations (azimuth increments past 360)
            self.opts["azimuth"] %= 360
            self.view_transition.set_origin(self.opts)
            target = self.get_camera_ground_intersection()
            target.setZ(self.opts["center"].z())
            self.view_transition.set_target(self.opts, elevation=90, azimuth=180, center=target)
            self.setMouseTracking(True)
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.view_transition.set_target(self.view_transition.origin_opts)
            self.view_transition.set_origin(self.opts)
            self.setMouseTracking(False)
            self.setCursor(Qt.CursorShape.ArrowCursor)

        self.mouse_position_2d_label.toggle(self._top_down_view_enabled)

        self.view_transition.start(dt_ms=16)

    def get_camera_ground_intersection(self):
        direction = self.view_movement.get_camera_forward_vector(
            elevation=self.opts["elevation"], azimuth=self.opts["azimuth"]
        )
        cam_pos = self.cameraPosition()
        # If there's ground to be seen (dir_z < 0)
        if direction.z() < 0:
            distance = -cam_pos.z() / direction.z()
            intersection_x = cam_pos.x() + distance * direction.x()
            intersection_y = cam_pos.y() + distance * direction.y()
            target_center = Vector(intersection_x, intersection_y, 0)
        else:
            target_center = self.opts["center"]

        return target_center

    def get_mouse_ground_intersection(self, mouse_pos=None):
        """
        Calculate the intersection of the mouse ray with the ground (z=0) plane
        
        Args:
            mouse_pos: Optional QPoint position. If None, uses the cursor position
        
        Returns:
            Vector: 3D world position where the mouse ray intersects the ground plane
            or None if no intersection
        """
        # Get mouse position
        if mouse_pos is None:
            screen_pos = self.mapFromGlobal(self.cursor().pos())
            mouse_x, mouse_y = screen_pos.x(), screen_pos.y()
        else:
            mouse_x, mouse_y = mouse_pos.x(), mouse_pos.y()
        
        # Get viewport dimensions
        width, height = self.width(), self.height()
        
        # Convert mouse position to normalized device coordinates (-1 to 1)
        ndc_x = (2.0 * mouse_x / width) - 1.0
        ndc_y = 1.0 - (2.0 * mouse_y / height)
        
        # Get camera properties
        projection = self.projectionMatrix(region=self.getViewport(), viewport=self.getViewport())  # Use current viewport
        view = self.viewMatrix()
        
        # Create combined inverse transform
        inv_transform = projection * view
        inv_transform = inv_transform.inverted()[0]  # inverted returns (matrix, success)
        
        # Create near and far points in NDC
        near_point = QVector4D(ndc_x, ndc_y, -1.0, 1.0)  # Near plane
        far_point = QVector4D(ndc_x, ndc_y, 1.0, 1.0)   # Far plane
        
        # Transform to world space
        near_world = inv_transform.map(near_point)
        far_world = inv_transform.map(far_point)
        
        # Perspective divide
        if near_world.w() != 0:
            near_world /= near_world.w()
        if far_world.w() != 0:
            far_world /= far_world.w()
        
        # Create ray from camera through mouse position
        ray_origin = Vector(near_world.x(), near_world.y(), near_world.z())
        ray_direction = Vector(
            far_world.x() - near_world.x(),
            far_world.y() - near_world.y(),
            far_world.z() - near_world.z()
        )
        ray_direction.normalize()
        
        # Calculate intersection with ground plane (z=0)
        if abs(ray_direction.z()) < 1e-6:  # Ray is parallel to ground
            return None
        
        t = -ray_origin.z() / ray_direction.z()
        if t < 0:  # Intersection is behind the camera
            return None
        
        # Calculate intersection point
        intersection = Vector(
            ray_origin.x() + ray_direction.x() * t,
            ray_origin.y() + ray_direction.y() * t,
            0.0  # On the ground plane
        )
        
        return intersection

    def mouseMoveEvent(self, ev):
        self.viewport_axes.update_rotation(
            elevation=self.opts["elevation"], azimuth=self.opts["azimuth"]
        )
        # For top-down view only panning and zoom are enabled
        if self._top_down_view_enabled:
            self._update_mouse_position_display(ev)
            self._top_down_mouse_drag_event(ev)
        else:
            return super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev) -> None:
        super().mousePressEvent(ev)
        # Initialize for top-down dragging
        self._last_mouse_drag_position = ev.position()

    def wheelEvent(self, ev):
        """Override wheel event to update text scaling after zoom."""
        if self._top_down_view_enabled and ev.position() is not None:
            self._update_mouse_position_display(ev)

        self._update_on_camera_change()
        super().wheelEvent(ev)

    def keyPressEvent(self, event):
        """Handle key press events for movement and other controls."""
        key = event.key()

        if self.view_movement.is_movement_key(key):
            # View movement
            self.view_movement.set_pressed_key(key)
            event.accept()
        elif key == Qt.Key.Key_T:
            # Top-down toggle
            self.toggle_top_down_view()
            event.accept()
        else:
            super().keyPressEvent(event)

        self._update_on_camera_change()

    def keyReleaseEvent(self, event):
        key = event.key()
        if self.view_movement.is_key_pressed(key):
            self.view_movement.unset_pressed_key(key)
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "viewport_axes"):
            self.viewport_axes.move(10, self.height() - self.viewport_axes.height() - 10)

    def update_data(self, rotation: Vector):
        roll, pitch, yaw = (rotation.x(), rotation.y(), rotation.z())
        transform = Transform3D()
        transform.rotate(roll, 1, 0, 0)
        transform.rotate(pitch, 0, 1, 0)
        transform.rotate(yaw, 0, 0, 1)
        self.world_origin.setTransform(transform)

    def move_car(self, x, y, heading_deg, steering_angle_deg):
        if hasattr(self, "car"):
            self.car.update(x, y, heading_deg, steering_angle_deg)

    def _top_down_mouse_drag_event(self, ev):
        if ev.buttons() == Qt.LeftButton:
            delta = ev.position().toPoint() - self._last_mouse_drag_position.toPoint()
            self._last_mouse_drag_position = ev.position()
            pan_speed = 0.003 * self.opts["distance"]
            dx = pan_speed * delta.x()
            dy = pan_speed * delta.y()
            self.pan(dy, dx, 0)

    def _view_movement_callback(self):
        speed = self.view_movement.get_speed(
            self.opts["elevation"], self.opts["azimuth"], self._top_down_view_enabled
        )
        self.setCameraPosition(pos=self.opts["center"] + speed)
        self._update_on_camera_change()

    def _animation_callback(self, center, distance, elevation, azimuth):
        self.setCameraPosition(pos=center, distance=distance, elevation=elevation, azimuth=azimuth)
        self._update_on_camera_change()

    def _update_on_camera_change(self):
        for item in self._on_camera_change_group:
            item.on_camera_change(self.opts)

    def _update_mouse_position_display(self, ev):
        pos = ev.position().toPoint()
        world_x, world_y = self._top_down_screen_to_world(pos.x(), pos.y())

        self.mouse_position_2d_label.update_coordinates(x=world_x, y=world_y)
        self.mouse_position_2d_label.update_position(x=pos.x(), y=pos.y())

    def _top_down_screen_to_world(self, screen_x, screen_y):
        viewport_width = self.width()
        viewport_height = self.height()
        aspect_ratio = viewport_width / viewport_height

        camera_center = self.opts["center"]
        camera_distance = self.opts["distance"] + camera_center.z()
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

    def _add_grids(self):
        grid_1m = GLGridItem()
        grid_1m.setSize(x=10, y=10, z=10)
        grid_1m.setSpacing(x=1, y=1, z=1)
        grid_1m.setColor((255, 255, 255, 30))
        self.addItem(grid_1m)

        grid_10cm = GLGridItem()
        grid_10cm.setSize(x=10, y=10, z=1)
        grid_10cm.setSpacing(x=0.1, y=0.1, z=0.1)
        grid_10cm.setColor((255, 255, 255, 15))
        self.addItem(grid_10cm)


# Demo
# -------------------------------------------------------------------------------------------------


class ControlPanel(QWidget):
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

        # Heading
        car_sliders_layout.addWidget(QLabel("Steering Angle:"), 3, 0)
        self.car_steering_angle_slider = QSlider(Qt.Horizontal)
        self.car_steering_angle_slider.setMinimum(-20)
        self.car_steering_angle_slider.setMaximum(+20)
        self.car_steering_angle_slider.setValue(0)
        car_sliders_layout.addWidget(self.car_steering_angle_slider, 3, 1)

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

        # Create floating control panel
        self.control_panel = ControlPanel()

        # Connect controls
        self.control_panel.roll_slider.valueChanged.connect(self._update_orientation)
        self.control_panel.pitch_slider.valueChanged.connect(self._update_orientation)
        self.control_panel.yaw_slider.valueChanged.connect(self._update_orientation)

        self.control_panel.animate_button.clicked.connect(self._toggle_animation)
        self.control_panel.reset_button.clicked.connect(self._reset_orientation)

        self.control_panel.car_x_slider.valueChanged.connect(self._update_car_position)
        self.control_panel.car_y_slider.valueChanged.connect(self._update_car_position)
        self.control_panel.car_heading_slider.valueChanged.connect(self._update_car_position)
        self.control_panel.car_steering_angle_slider.valueChanged.connect(self._update_car_position)

        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation)
        self.animation_angle = 0

        # Add control panel toggle action
        toggle_controls_action = self.menuBar().addAction("Toggle Controls")
        toggle_controls_action.triggered.connect(self._toggle_control_panel)
        toggle_controls_action.setShortcut("Ctrl+C")

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
        rotation = Vector(
            self.control_panel.roll_slider.value(),
            self.control_panel.pitch_slider.value(),
            self.control_panel.yaw_slider.value(),
        )
        self.map_3d.update_data(rotation)

    def _update_car_position(self):
        """Update car position based on slider values."""
        x = self.control_panel.car_x_slider.value() / 10.0
        y = self.control_panel.car_y_slider.value() / 10.0
        heading = self.control_panel.car_heading_slider.value()
        steering_angle = self.control_panel.car_steering_angle_slider.value()
        self.map_3d.move_car(x, y, heading, steering_angle)

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
        rotation = Vector(
            45 * np.sin(np.deg2rad(self.animation_angle)),
            45 * np.sin(np.deg2rad(self.animation_angle + 120)),
            self.animation_angle,
        )

        # Update slider positions
        self.control_panel.roll_slider.setValue(int(rotation.x()))
        self.control_panel.pitch_slider.setValue(int(rotation.y()))
        self.control_panel.yaw_slider.setValue(int(rotation.z()))

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
