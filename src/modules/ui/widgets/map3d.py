from enum import Enum
import sys

sys.path.append("C:/Users/ihazu/Desktop/projects/toysim_server/src")
import numpy as np
from typing import Any, Iterable, Set
from copy import deepcopy

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QVector4D, QColor
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

from modules.ui.presets import Fonts, GLColors, UIColors

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
            f"""
            background-color: rgba{QColor(UIColors.FOREGROUND).getRgb()}; 
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
        # Distance from cursor
        x += x_offset
        y += y_offset

        # Within widget bounds
        if check_parent_bounds:
            if x + self.width() > self.parent().width():
                x = x - self.width() - 5
            if y < 0:
                y = y + 15

        self.move(x, y)


class ReferenceFrameLabel(QLabel):
    """Label showing the active reference frame with enhanced styling"""

    NAMES_TO_ICONS = {"World": "🌍", "Vehicle": "🚗", "Camera": "📷", "DEFAULT": "📐"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet(
            f"""
            background-color: rgba{QColor(UIColors.FOREGROUND).getRgb()}; 
            color: rgba{GLColors.ORANGE.getRgb()}; 
            padding: 5px;
            border-radius: 5px;
            border: 1px solid rgba{QColor(UIColors.ON_FOREGROUND_DIM).getRgb()};
            font-weight: bold;
        """
        )
        self.setFont(Fonts.OpenGLMonospace)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.show()

    def set_name(self, name: str):
        icon = self.NAMES_TO_ICONS.get(name, self.NAMES_TO_ICONS["DEFAULT"])
        self.setText(f"{icon} {name}")
        self.adjustSize()

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
        if self._active and self._active != name:
            self._frames_by_name[self._active].set_focus(False)
        self._active = name
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

    def on_camera_change(self, elevation, azimuth, init_transform):
        self.update_rotation(elevation, azimuth, init_transform)

    def update_rotation(self, elevation: float, azimuth: float, init_transform=Transform3D()):
        self.rotation = Transform3D()
        self.rotation.rotate(-elevation, 1, 0, 0)  # Pitch
        self.rotation.rotate(azimuth, 0, 1, 0)  # Yaw

        rf_rotation = Transform3D()
        # From Rz -> -Ry
        init_matrix = init_transform.matrix()
        rf_rotation.setColumn(0, QVector4D(init_matrix[0, 0], 0, init_matrix[1, 0], 0))
        rf_rotation.setColumn(1, QVector4D(0, 1, 0, 0))
        rf_rotation.setColumn(2, QVector4D(init_matrix[0, 1], 0, init_matrix[1, 1], 0))
        rf_rotation.setColumn(3, QVector4D(0, 0, 0, 1))

        self.rotation = self.rotation * rf_rotation 

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
            ("X", Vector(0, 0, -1), GLColors.RED),
            ("Y", Vector(1, 0, 0), GLColors.GREEN),
            ("Z", Vector(0, 1, 0), GLColors.BLUE),
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


class Roadmark:
    def __init__(self, position):
        self.position = position
    
    def get_position(self):
        return self.position

class NavigationData:
    def __init__(self, parent_widget: GLViewWidget):
        self.parent_widget = parent_widget
        self.roadmarks = []

        self.roadmarks_scatter = GLScatterPlotItem(size=10, color=GLColors.TURQUOIS.getRgbF())
        self.roadmarks_scatter.setDepthValue(1)
        self.parent_widget.addItem(self.roadmarks_scatter)

        self.path = GLLinePlotItem(color=GLColors.TURQUOIS.getRgbF(), width=3, antialias=True)
        self.parent_widget.addItem(self.path)

    def update(self, roadmarks: np.ndarray, path: np.ndarray):
        roadmarks3d = np.column_stack((roadmarks, np.zeros(roadmarks.shape[0]) + 0.001))
        self.roadmarks_scatter.setData(pos=roadmarks3d)
        path3d = np.column_stack((path, np.zeros(path.shape[0])))
        self.path.setData(pos=path3d)
        self.roadmarks = [Roadmark(Vector(x)) for x in roadmarks3d]
        return self.roadmarks

    def _clear(self):
        for rm in self.roadmarks:
            self.parent_widget.removeItem(rm)
        self.roadmarks = []
        

class PositionTracker:
    def __init__(self, parent: GLViewWidget):
        self.parent = parent

        self._items = []
        self._cache = []

        self._rf = None
        self._rf_position = None
        self._T_world2rf = None

        self.distance_color = GLColors.PURPLE
        self.position_color = GLColors.TURQUOIS

    def update_rf(self, rf: ReferenceFrame):
        self._rf = rf
        self._rf_position = self._rf.get_position()   
        self._T_world2rf, b = self._rf.get_transform().inverted()
        assert b, f"Failed to invert {self._rf.get_transform()}"

    def update_items(self, items):
        self._items = items
        self.update_tracking()

    def update_tracking(self):
        cache_index = 0
        for item in self._items:
            self._create_if_not_cached(cache_index)
            self._update_cached_item_data(cache_index, self._rf_position, item.get_position())
            cache_index += 1
        self._hide_unused_cached_items(cache_index)

    def _create_if_not_cached(self, cache_index: int):
        if cache_index == len(self._cache):
            line = self._create_item_distance_line()
            position = self._create_item_label()
            self._cache.append((line, position))

    def _update_cached_item_data(self, cache_index: int, rf_pos_world: Vector, item_pos_world: Vector):
        line, position = self._cache[cache_index]
        self._update_item_distance_line(line, rf_pos_world, item_pos_world)
        self._update_item_label(position, rf_pos_world, item_pos_world)

    def _hide_unused_cached_items(self, cache_index: int):
        for i in range(cache_index, len(self._cache)):
            for item in self._cache[i]:
                item.hide()

    def _create_item_distance_line(self) -> GLLinePlotItem:
        line = GLLinePlotItem(color=self.distance_color.getRgbF(), antialias=True)
        self.parent.addItem(line)
        return line

    def _update_item_distance_line(self, line: GLLinePlotItem, start: Vector, end: Vector):
        line.setData(pos=(start, end))
        line.show()

    def _create_item_label(self) -> GLMultiTextItem:
        label = GLMultiTextItem(font=Fonts.OpenGLMonospace)
        self.parent.addItem(label)
        return label

    def _update_item_label(self, label: GLMultiTextItem, pos_rf_world: Vector, pos_item_world: Vector):
        dist_rf = pos_item_world.distanceToPoint(pos_rf_world)
        pos_rf = self._T_world2rf.map(pos_item_world)
        pos_text_pos = Vector(pos_item_world + Vector(0, 0, 0.05))
        label.setData(
            pos=pos_text_pos,
            colors=[self.position_color, self.distance_color],
            texts=[
                f"({pos_rf.x():.3f}, {pos_rf.y():.3f}, {pos_rf.z():.3f})",
                f" {dist_rf:.3f}",
            ],
        )
        label.show()


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
        self.setBackgroundColor(UIColors.FOREGROUND)
        self.setCameraPosition(
            pos=self.INIT_OPTS["center"],
            distance=self.INIT_OPTS["distance"],
            elevation=self.INIT_OPTS["elevation"],
            azimuth=self.INIT_OPTS["azimuth"],
        )
        # self.setFocusPolicy(Qt.StrongFocus)  # Accept (keyboard) events without focus

        # Items
        self._add_grids()
        self.world_rf = ReferenceFrame(parent_widget=self, name="World")
        self.viewport_axes = ViewportAxes(self)
        self.car = Car3D(parent_widget=self)
        self.mouse_position_2d_label = PositionLabel2D(parent=self)
        self.navigation_data = NavigationData(parent_widget=self)

        ref_frames = [self.world_rf, self.car.car_rf, self.car.camera_rf]
        self.rfs = ReferenceFrames(parent=self, ref_frames=ref_frames)

        # Item position tracking
        self.path_planner_position_tracker = PositionTracker(parent=self)
        self.path_planner_position_tracker.update_rf(self.rfs.get_active())
        self.rf_position_tracker = PositionTracker(parent=self)
        self.rf_position_tracker.position_color = GLColors.WHITE
        self.rf_position_tracker.update_rf(self.rfs.get_active())
        self.rf_position_tracker.update_items(items=ref_frames)

        # Views
        self.view_transition = ViewTransition(animation_callback=self._animation_callback)
        self.view_manager = ViewManager(parent_widget=self, view_transition=self.view_transition)

        # Movement
        self.view_movement = ViewMovement(apply_movement_callback=self._view_movement_callback)

        # Events
        self._on_camera_change_group: Set[Any] = {self.viewport_axes}
        self._update_on_camera_change()

    def update_data(self, car_x, car_y, car_heading, car_steering_angle, roadmarks, path):
        items = self.navigation_data.update(roadmarks=roadmarks, path=path)
        self.car.update(x=car_x, y=car_y, heading_deg=car_heading, steering_angle_deg=car_steering_angle)

        self.path_planner_position_tracker.update_items(items=items)
        self.rf_position_tracker.update_tracking()

    def update_car_data(self, x, y, heading, steering_angle):
        self.car.update(x, y, heading, steering_angle)
        self.rf_position_tracker.update_tracking()
        self.path_planner_position_tracker.update_tracking()

    def toggle_top_down_view(self):
        """Toggle between top-down view and normal 3D view with animation."""
        self.view_manager.toggle(ViewManager.Views.TOPDOWN)
        self.mouse_position_2d_label.toggle(self.view_manager.is_active(ViewManager.Views.TOPDOWN))

    def mouseMoveEvent(self, ev):
        self._update_on_camera_change()
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
        active_rf = self.rfs.get_active()
        self.rf_position_tracker.update_rf(active_rf)
        self.path_planner_position_tracker.update_rf(active_rf)

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
            tr = Transform3D(self.rfs.get_active().get_transform())
            item.on_camera_change(
                elevation=self.opts["elevation"],
                azimuth=self.opts["azimuth"],
                init_transform=tr,
            )

    def _update_mouse_position_display(self, ev):
        pos = ev.position().toPoint()
        world_x, world_y = self._top_down_screen_to_world(pos.x(), pos.y())

        # Transform from world to rf
        rf = self.rfs.get_active()
        tr_world2rf, b = rf.get_transform().inverted()
        assert b, f"Transform matrix of the '{rf.name}' reference frame could not be inverted"
        world_point = Vector(world_x, world_y, 0)
        rf_point = tr_world2rf.map(world_point)
        
        self.mouse_position_2d_label.update_coordinates(x=rf_point.x(), y=rf_point.y())
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

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.map_3d = Map3D()
        main_layout.addWidget(self.map_3d)

        self.control_panel = ControlPanel()
        self.control_panel.roll_slider.valueChanged.connect(self._update_orientation)
        self.control_panel.pitch_slider.valueChanged.connect(self._update_orientation)
        self.control_panel.yaw_slider.valueChanged.connect(self._update_orientation)

        self.control_panel.animate_button.clicked.connect(self._toggle_animation)
        self.control_panel.reset_button.clicked.connect(self._reset_orientation)

        self.control_panel.car_x_slider.valueChanged.connect(self._update_car_position)
        self.control_panel.car_y_slider.valueChanged.connect(self._update_car_position)
        self.control_panel.car_heading_slider.valueChanged.connect(self._update_car_position)
        self.control_panel.car_steering_angle_slider.valueChanged.connect(self._update_car_position)

        toggle_controls_action = self.menuBar().addAction("Toggle Controls")
        toggle_controls_action.triggered.connect(self._toggle_control_panel)
        toggle_controls_action.setShortcut("Ctrl+C")

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

    def _reset_orientation(self):
        """Reset to default orientation."""
        self.control_panel.roll_slider.setValue(0)
        self.control_panel.pitch_slider.setValue(0)
        self.control_panel.yaw_slider.setValue(0)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Map3DDemo()
    window.show()
    sys.exit(app.exec())
