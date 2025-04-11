from enum import Enum
import sys

sys.path.append("C:/Users/ihazu/Desktop/projects/toysim_server/src")
import numpy as np
from typing import Any, Iterable, Set
from copy import deepcopy

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen
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
    QSizePolicy,
)

from pyqtgraph.opengl import GLViewWidget, GLGridItem, GLScatterPlotItem, GLLinePlotItem
from pyqtgraph import Transform3D, Vector

from modules.ui.presets import Fonts, MColors

from modules.ui.widgets.opengl.shapes import OpaqueCylinder
from modules.ui.widgets.opengl.models import Car3D
from modules.ui.widgets.opengl.helpers import (
    ReferenceFrame,
    GLMultiTextItem,
    get_camera_forward_vector,
    get_camera_right_vector,
    get_camera_ground_intersection,
)


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


class ReferenceFrameLabel(QLabel):
    """Label showing the active reference frame with enhanced styling"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        color = (
            MColors.ORANGE.red(),
            MColors.ORANGE.green(),
            MColors.ORANGE.blue(),
            MColors.ORANGE.alpha(),
        )
        self.setStyleSheet(
            f"""
            background-color: rgba(0, 0, 0, 180); 
            color: rgba{color}; 
            padding: 5px;
            border-radius: 5px;
            border: 1px solid rgba(120, 120, 120, 50);
            font-weight: bold;
        """
        )
        self.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumWidth(100)
        self.show()

    def set_name(self, name: str):
        self.setText(f"📐 {name}")


class ReferenceFrames:
    def __init__(self, parent: GLViewWidget, ref_frames: Iterable[ReferenceFrame] = []):
        self.parent = parent

        self._frames_by_name = dict()
        self._names_by_key = dict()
        self._active: str = None

        self._keys = [
            Qt.Key.Key_1,
            Qt.Key.Key_2,
            Qt.Key.Key_3,
            Qt.Key.Key_4,
            Qt.Key.Key_5,
            Qt.Key.Key_6,
        ]

        self.active_label = ReferenceFrameLabel(parent=self.parent)
        self.update_label_position()

        for rf in ref_frames:
            self.add(rf)
        if len(ref_frames):
            self.set_active(ref_frames[0].name)

    def is_rf_key(self, key: Qt.Key):
        return key in self._keys

    def set_active_rf_by_key(self, key):
        name = self._names_by_key.get(key, None)
        if name is not None: 
            self.set_active(name)

    def add(self, rf: ReferenceFrame):
        assert len(self._frames_by_name) <= len(
            self._keys
        ), "Maximum number of reference frames reached, update the _keys list to add more"
        assert self._frames_by_name.get(rf.name) is None, f"Ref frame {rf.name} already defined"

        key_index = len(self._frames_by_name)
        key = self._keys[key_index]
        self._names_by_key[key] = rf.name
        self._frames_by_name[rf.name] = rf

    def remove(self, name: str):
        del self._frames_by_name[name]

    def set_active(self, name: str):
        self._active = name
        for frame in self._frames_by_name.values():
            frame.set_focus(False)
        self._frames_by_name[name].set_focus(True)
        self.active_label.set_name(name)

    def get_active(self) -> ReferenceFrame:
        return self._frames_by_name[self._active]

    def get_frames(self) -> Iterable[ReferenceFrame]:
        return self._frames_by_name.values()

    def update_label_position(self, x=10, y=-10):
        self.active_label.move(x, self.parent.height() - self.active_label.height() + y)


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

    def update_position(self, x=10, y=-10):
        self.move(x, self.parent().height() - self.height() + y)

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


class ViewManager:
    class Views(Enum):
        DEFAULT = 0
        TOPDOWN = 1

    def __init__(self, parent_widget: GLViewWidget, view_transition: ViewTransition):
        self.parent_widget = parent_widget
        self.view_transition = view_transition
        self.active_view = self.Views.DEFAULT

    def is_active(self, view: Views):
        return self.active_view == view

    def toggle(self, view: Views):
        self.active_view = self.Views.DEFAULT if self.active_view == view else view

        # Cancel ongoing animation
        if self.view_transition.is_transitioning():
            self.view_transition.stop()

        if self.is_active(self.Views.TOPDOWN):
            self._set_topdown()
        else:
            self._set_default()

        self.view_transition.start(dt_ms=16)

    def _set_topdown(self):
        opts = self.parent_widget.opts
        opts["azimuth"] %= 360
        self.view_transition.set_origin(opts)

        cam_position = self.parent_widget.cameraPosition()
        cam_direction = get_camera_forward_vector(opts["elevation"], opts["azimuth"])
        ground_position = get_camera_ground_intersection(cam_position, cam_direction)
        if ground_position is None:
            target = opts["center"]
        else:
            target = Vector(ground_position.x(), ground_position.y(), opts["center"].z())
        self.view_transition.set_target(opts, elevation=90, azimuth=180, center=target)

        self.parent_widget.setMouseTracking(True)
        self.parent_widget.setCursor(Qt.CursorShape.CrossCursor)

    def _set_default(self):
        self.view_transition.set_target(self.view_transition.origin_opts)
        self.view_transition.set_origin(self.parent_widget.opts)

        self.parent_widget.setMouseTracking(False)
        self.parent_widget.setCursor(Qt.CursorShape.ArrowCursor)


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
        forward = get_camera_forward_vector(elevation, azimuth)
        if topdown_view:
            # Make W/S pan up/down
            forward.setX(-forward.z())
            forward.setZ(0)
        right = get_camera_right_vector(azimuth)
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


class NavigationData:
    def __init__(self, parent_widget: GLViewWidget):
        self.parent_widget = parent_widget
        self.roadmarks = []

        self.roadmarks_scatter = GLScatterPlotItem(size=10, color=MColors.TURQUOIS.getRgbF())
        self.roadmarks_scatter.setDepthValue(1)
        self.parent_widget.addItem(self.roadmarks_scatter)

        self.path = GLLinePlotItem(color=MColors.TURQUOIS.getRgbF(), width=3, antialias=True)
        self.parent_widget.addItem(self.path)

    def update(self, roadmarks: np.ndarray, path: np.ndarray):
        self._clear()
        for rm in roadmarks:
            mesh = OpaqueCylinder(
                radius=0.02,
                height=0.001,
                color=MColors.GRAY_TRANS.getRgbF(),
                drawEdges=False,
                drawFaces=True,
            )
            mesh.setDepthValue(-1)
            mesh.translate(dx=rm[0], dy=rm[1], dz=0.000)
            self.parent_widget.addItem(mesh)
            self.roadmarks.append(mesh)
        roadmarks3d = np.column_stack((roadmarks, np.zeros(roadmarks.shape[0]) + 0.001))
        self.roadmarks_scatter.setData(pos=roadmarks3d)
        path3d = np.column_stack((path, np.zeros(path.shape[0])))
        self.path.setData(pos=path3d)
        return self.roadmarks

    def _clear(self):
        for rm in self.roadmarks:
            self.parent_widget.removeItem(rm)
        self.roadmarks = []


class NavigationDataPlayer:
    def __init__(self, update_data_callback: callable):
        self.update_data_callback = update_data_callback

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
        # self.update_data_callback(rm_data.roadmarks, rm_data.path)
        self.update_data_callback(np.array([[1, 0]]), np.array([]))
        self.idx = (self.idx + 1) % len(self.data)


class PositionTracker:
    def __init__(self, parent: GLViewWidget):
        self.parent = parent
        self.items_visual_clues = set()

    def add_item(self, item):
        self.items_visual_clues.add(item)
        self.parent.addItem(item)

    def remove_item(self, item):
        self.parent.removeItem(item)
        self.items_visual_clues.remove(item)

    def clear_items(self):
        for item in self.items_visual_clues:
            self.parent.removeItem(item)
        self.items_visual_clues.clear()

    def update_rf(self, transform: Vector):
        self._rf_transform = transform
        self._rf_position = Vector(transform.matrix()[:3, 3])

    def update_items(self, items):
        """Clears items and adds new"""
        self.clear_items()
        for item in items:
            p_world = item.get_position()
            p_active_rf = self._rf_transform.map(p_world)
            origin_distance = p_world.distanceToPoint(self._rf_position)
            self._add_item_distance_line(self._rf_position, p_world)
            self._add_item_label(p_world, p_active_rf, origin_distance)

    def update(self, rf_transform: Vector, items):
        self.update_rf(rf_transform)
        self.update_items(items)

    def _add_item_distance_line(self, start: Vector, end: Vector):
        line = GLLinePlotItem(color=MColors.PURPLISH_LIGHT.getRgbF(), antialias=True)
        line.setData(pos=(start, end))
        self.add_item(line)

    def _add_item_label(
        self, pos_world: Vector, pos_rf: Vector, dist_rf: float
    ):
        pos_text_pos = Vector(pos_world + Vector(0, 0, 0.05))
        multi_label = GLMultiTextItem(pos=pos_text_pos, font=Fonts.Monospace)
        multi_label.setData(
            colors=[MColors.TURQUOIS, MColors.PURPLISH_LIGHT],
            texts=[
                f"({pos_rf.x():.3f}, {pos_rf.y():.3f}, {pos_rf.z():.3f})",
                f" {dist_rf:.3f}",
            ],
        )
        self.add_item(multi_label)


class Map3D(GLViewWidget):
    INIT_OPTS = {
        "center": Vector(-1.60, 1.5, 1.8),
        "distance": 0.35,
        "elevation": 35,
        "azimuth": 145,
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
        self.world_rf = ReferenceFrame(parent_widget=self, name="World")
        self.viewport_axes = ViewportAxes(self)
        self.car = Car3D(parent_widget=self)
        self.mouse_position_2d_label = PositionLabel2D(parent=self)
        self.navigation_data = NavigationData(parent_widget=self)
        self.navigation_data_player = NavigationDataPlayer(self.update_navigation_data)

        ref_frames = [self.world_rf, self.car.car_rf, self.car.camera_rf]
        self.rfs = ReferenceFrames(parent=self, ref_frames=ref_frames)

        # Item position tracking
        self.path_planner_position_tracker = PositionTracker(parent=self)
        self.rf_position_tracker = PositionTracker(parent=self)
        self.rf_position_tracker.update_rf(self.rfs.get_active().get_transform())
        self.rf_position_tracker.update_items(items=ref_frames)

        # Views
        self.view_transition = ViewTransition(animation_callback=self._animation_callback)
        self.view_manager = ViewManager(parent_widget=self, view_transition=self.view_transition)

        # Movement
        self.view_movement = ViewMovement(apply_movement_callback=self._view_movement_callback)

        # Events
        self._on_camera_change_group: Set[Any] = {self.viewport_axes}
        self._update_on_camera_change()

    def update_navigation_data(self, *args, **kwargs):
        items = self.navigation_data.update(*args, **kwargs)
        origin_transform = self.rfs.get_active().get_transform()
        self.path_planner_position_tracker.update(rf_transform=origin_transform, items=items)

    def update_car_data(self, x, y, heading, steering_angle):
        self.car.update(x, y, heading, steering_angle)
        self.rf_position_tracker.update_items(items=self.rfs.get_frames())

    def toggle_top_down_view(self):
        """Toggle between top-down view and normal 3D view with animation."""
        self.view_manager.toggle(ViewManager.Views.TOPDOWN)
        self.mouse_position_2d_label.toggle(self.view_manager.is_active(ViewManager.Views.TOPDOWN))

    def mouseMoveEvent(self, ev):
        self.viewport_axes.update_rotation(self.opts["elevation"], self.opts["azimuth"])
        if self.view_manager.is_active(ViewManager.Views.TOPDOWN):
            self._update_mouse_position_display(ev)
        else:
            return super().mouseMoveEvent(ev)

    def wheelEvent(self, ev):
        if self.view_manager.is_active(ViewManager.Views.TOPDOWN) and ev.position() is not None:
            self._update_mouse_position_display(ev)

        self._update_on_camera_change()
        super().wheelEvent(ev)

    def keyPressEvent(self, event):
        if self._custom_keypress_event(event):
            event.accept()
        else:
            super().keyPressEvent(event)

        self._update_on_camera_change()

    def _custom_keypress_event(self, event) -> bool:
        """Custom key maps
        Returns:
            bool: True if the key was handled by this function, False otherwise.
        """

        key = event.key()
        if self.view_movement.is_movement_key(key):
            self.view_movement.set_pressed_key(key)
        elif self.rfs.is_rf_key(key):
            self.rfs.set_active_rf_by_key(key)
            self._on_ref_frame_change()
        elif key == Qt.Key.Key_T:
            self.toggle_top_down_view()
        else:
            return False
        return True

    def _on_ref_frame_change(self):
        self.rf_position_tracker.update_rf(self.rfs.get_active().get_transform())

    def keyReleaseEvent(self, event):
        key = event.key()
        if self.view_movement.is_key_pressed(key):
            self.view_movement.unset_pressed_key(key)
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.viewport_axes.update_position()
        self.rfs.update_label_position()

    def _view_movement_callback(self):
        speed = self.view_movement.get_speed(
            self.opts["elevation"],
            self.opts["azimuth"],
            self.view_manager.is_active(ViewManager.Views.TOPDOWN),
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

        # Correct for reference frame position
        # TODO: this only takes into account position, not rotation
        ref_frame_offset = self.rfs.get_active().get_position()
        ref_frame_x = world_x - ref_frame_offset.x()
        ref_frame_y = world_y - ref_frame_offset.y()

        self.mouse_position_2d_label.update_coordinates(x=ref_frame_x, y=ref_frame_y)
        self.mouse_position_2d_label.update_position(x=pos.x(), y=pos.y())

    def _top_down_screen_to_world(self, screen_x, screen_y):
        viewport_width = self.width()
        viewport_height = self.height()
        aspect_ratio = viewport_width / viewport_height

        # <-1, 1> Normalized
        norm_screen_x = (2.0 * screen_x / viewport_width) - 1.0
        norm_screen_y = 1.0 - (2.0 * screen_y / viewport_height)

        cam_center = self.opts["center"]
        cam_distance = self.opts["distance"] + cam_center.z()
        cam_fov = self.opts["fov"]

        # Visible ground plane area
        ground_plane_height = 2.0 * cam_distance * np.tan(np.deg2rad(cam_fov / 2))
        ground_plane_width = ground_plane_height

        # World (X, Y) are Viewport (Y, -X)
        world_x = (
            (cam_center.x() * aspect_ratio) + (norm_screen_y * ground_plane_height / 2.0)
        ) / aspect_ratio
        world_y = cam_center.y() - (norm_screen_x * ground_plane_width / 2.0)

        return world_x, world_y

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
        self.map_3d.update_car_data(x, y, heading, steering_angle)

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
