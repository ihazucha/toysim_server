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
)

from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLGridItem, GLTextItem, GLBoxItem
from pyqtgraph import Transform3D

from modules.ui.plots import Colors
from datalink.data import Rotation


class Car3D:
    """Simple 3D car model for the Map3D visualization."""
    
    def __init__(self, parent_widget: GLViewWidget):
        """Create a simple car model starting at origin."""
        self.parent_widget = parent_widget
        
        self.length = 0.185
        self.width = 0.155 
        self.height = 0.2
        wheelbase = 0.185
        track = 0.155
        wheel_radius = 0.06
        wheel_width = 0.04

        # Create car body (main chassis)
        self.body = GLBoxItem(size=QVector3D(self.length, self.width, self.height/1.5))
        self.body.setColor((175, 0, 0, 200))  # Red, slightly transparent
        # Store initial offsets instead of translating directly
        self.body_offset = QVector3D(0, -self.width/2, wheel_radius/2)
        
        # Create wheels (4 black cylinders)
        self.wheels = []
        self.wheel_offsets = []

        
        # Wheel positions relative to car center
        wheel_positions = [
            (self.length - wheel_radius/2, self.width/2, -wheel_radius/2),  # Front Left
            (self.length - wheel_radius/2, -self.width/2 - wheel_width, -wheel_radius/2),  # Front Right
            (-wheel_radius/2, self.width/2, -wheel_radius/2),  # Rear Left
            (-wheel_radius/2, -self.width/2 - wheel_width, -wheel_radius/2)  # Rear Right
        ]
        
        for wx, wy, wz in wheel_positions:
            wheel = GLBoxItem(size=QVector3D(wheel_radius, wheel_width, wheel_radius))
            wheel.setColor((125, 125, 125, 255))  # Black
            # Store wheel offset instead of translating directly
            self.wheel_offsets.append(QVector3D(wx, wy, wz + wheel_radius/2))
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

class BasisVectors3D:
    BASE_FONT_SIZE = 16
    MIN_FONT_SIZE = 6
    FONT = QFont("Consolas, Helvetica", BASE_FONT_SIZE)

    def __init__(self, parent_widget: GLViewWidget, name: str):
        self.x = GLLinePlotItem(pos=np.array([[0] * 3, [0.05, 0, 0]]), color=(1, 0, 0, 1), width=3)
        self.y = GLLinePlotItem(pos=np.array([[0] * 3, [0, 0.05, 0]]), color=(0, 1, 0, 1), width=3)
        self.z = GLLinePlotItem(pos=np.array([[0] * 3, [0, 0, 0.05]]), color=(0, 0, 1, 1), width=3)
        self.xyz_lines = [self.x, self.y, self.z]

        self.x_label = GLTextItem(pos=np.array([0.06, 0, 0]), text=f"X({name})", color=(255, 0, 0, 255))
        self.y_label = GLTextItem(pos=np.array([0, 0.06, 0]), text=f"Y({name})", color=(0, 255, 0, 255))
        self.z_label = GLTextItem(pos=np.array([0, 0, 0.06]), text=f"Z({name})", color=(0, 0, 255, 255))
        self.xyz_labels = [self.x_label, self.y_label, self.z_label]

        self._add_to_parent_widget(parent_widget)

    def _add_to_parent_widget(self, widget: GLViewWidget):
        for line in self.xyz_lines:
            widget.addItem(line)
        for label in self.xyz_labels:
            widget.addItem(label)

    def scale_font_by_camera_distance(self, distance: float):
        scale = 2.5
        size = self.BASE_FONT_SIZE / (distance * scale)
        size = max(self.MIN_FONT_SIZE, size)
        for label in self.xyz_labels:
            self.FONT.setPointSizeF(size)
            label.setData(font=self.FONT)

    def transform(self, tr: Transform3D):
        for line in self.xyz_lines:
            line.setTransform(tr)
        for label in self.xyz_labels:
            label.setTransform(tr)


class ViewportAxes(QWidget):
    """Fixed coordinate system display for the bottom left corner that rotates with camera."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(150, 150)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.rotation = Transform3D()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Center
        start_x = self.width() / 2
        start_y = self.height() / 2
        
        axis_length = 50
        
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
            pen = QPen(color=color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(start_x, start_y, end_x, end_y)
                        
            # Label
            font = painter.font()
            font.setBold(True)
            font.setFamily("Consolas, Helvetica")
            painter.setFont(font)
            painter.drawText(end_x + 2, end_y + 2, label)
        
    def update_rotation(self, glview_widget):
        """Update rotation based on camera's view matrix."""
        elevation = glview_widget.opts['elevation']
        azimuth = glview_widget.opts['azimuth']
        
        self.rotation = Transform3D()
        self.rotation.rotate(azimuth, 0, 1, 0) # Yaw
        self.rotation.rotate(elevation, 1, 0, 0) # Pitch
        self.update()
        

class Map3D(GLViewWidget):
    INIT_POS = QVector3D(0, 0, 0)
    INIT_DIST = 5
    INIT_AZIMUTH = 135

    def __init__(self):
        super().__init__()
        self.setBackgroundColor(Colors.FOREGROUND)
        self.setCameraPosition(
            pos=Map3D.INIT_POS, distance=Map3D.INIT_DIST, azimuth=Map3D.INIT_AZIMUTH
        )
        # React to shortcuts without focus
        self.setFocusPolicy(Qt.StrongFocus)

        # Reference frames
        self.world_basis_vectors = BasisVectors3D(parent_widget=self, name="w")
        self.world_basis_vectors.scale_font_by_camera_distance(distance=Map3D.INIT_DIST)
        
        # Add viewport axes in bottom left corner
        self.viewport_axes = ViewportAxes(self)
        self.viewport_axes.move(10, self.height() - self.viewport_axes.height() - 10)
        self.viewport_axes.update_rotation(self)
        self.viewport_axes.show()
        
        
        # Create car model
        self.car = Car3D(parent_widget=self)

        self._add_grid()

        self._top_down_view_enabled = False
        self._stored_camera_state = None

        # Create coordinate display label
        from PySide6.QtWidgets import QLabel

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

        # Create center cursor
        self.center_cursor = QWidget(self)
        self.center_cursor.setFixedSize(20, 20)
        self.center_cursor.setAttribute(Qt.WA_TranslucentBackground)
        self.center_cursor.paintEvent = self._paint_center_cursor
        self.center_cursor_enabled = False
        self.center_cursor.hide()

        # Center coordinates label
        self.center_coordinates_label = QLabel(self)
        self.center_coordinates_label.setStyleSheet(
            """
            background-color: rgba(0, 0, 0, 120); 
            color: white; 
            padding: 5px;
            border-radius: 3px;
            """
        )
        self.center_coordinates_label.setAlignment(Qt.AlignCenter)
        self.center_coordinates_label.setMinimumWidth(150)
        self.center_coordinates_label.hide()

    # Add methods to control the car
    def move_car(self, x, y, heading=None):
        """Move the car to a new position and optionally change heading."""
        if hasattr(self, 'car'):
            # Keep current heading if not specified
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
            # Show center cursor in top-down view
            self._update_center_cursor()
            self.center_cursor.show()
            self.center_coordinates_label.show()
        else:
            self._restore_camera_state()
            self.setMouseTracking(False)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.coordinate_label.hide()
            # Hide center cursor when not in top-down view
            self.center_cursor.hide()
            self.center_coordinates_label.hide()

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
        # For top-down view only panning and zoom are enabled
        self.viewport_axes.update_rotation(self)
        if self._top_down_view_enabled:
            # Update coordinate display
            self._update_coordinate_display(ev)

            if ev.buttons() == Qt.LeftButton:
                delta = ev.position().toPoint() - self._last_mouse_position.toPoint()
                self._last_mouse_position = ev.position()
                pan_speed = 0.02
                dx = pan_speed * delta.x()
                dy = pan_speed * delta.y()
                self.pan(dy, dx, 0)
                # After panning, update center cursor coordinates
                self._update_center_cursor()
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
            self._update_coordinate_display(ev)
        if hasattr(self, "world_basis_vectors"):
            self.world_basis_vectors.scale_font_by_camera_distance(self.opts["distance"])
        # Update center cursor coordinates when zooming
        if self._top_down_view_enabled:
            self._update_center_cursor()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_T:
            self.toggle_top_down_view()
            event.accept()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        """Handle resize to reposition viewport axes in bottom left corner."""
        super().resizeEvent(event)
        if hasattr(self, 'viewport_axes'):
            self.viewport_axes.move(10, self.height() - self.viewport_axes.height() - 10)
        # Update center cursor position on resize
        if self._top_down_view_enabled:
            self._update_center_cursor()

    def _update_coordinate_display(self, ev):
        """Calculate and display world coordinates from mouse position."""
        pos = ev.position().toPoint()
        world_x, world_y = self._screen_to_world(pos.x(), pos.y())

        self.coordinate_label.setText(f"X: {world_x:.3f}, Y: {world_y:.3f}")
        
        label_x = pos.x() + 15
        label_y = pos.y() - 30
        
        # Stay within widget bounds
        if label_x + self.coordinate_label.width() > self.width():
            label_x = pos.x() - self.coordinate_label.width() - 5
        if label_y < 0:
            label_y = pos.y() + 15

        self.coordinate_label.move(label_x, label_y)

    def _screen_to_world(self, screen_x, screen_y):
        """
        Convert screen coordinates to world coordinates in top-down view,
        properly accounting for camera projection and FOV.
        """
        viewport_width = self.width()
        viewport_height = self.height()
        aspect_ratio = viewport_width / viewport_height
        
        camera_distance = self.opts["distance"]
        camera_center = self.opts["center"]
        fov = self.opts.get("fov", 60)
        
        # Visible ground plane area
        visible_height = 2.0 * camera_distance * np.tan(np.deg2rad(fov/2))
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

    def _paint_center_cursor(self, event):
        """Paint crosshair cursor at center of widget."""
        painter = QPainter(self.center_cursor)
        painter.setRenderHint(QPainter.Antialiasing)
        
        size = self.center_cursor.width()
        center = size // 2
        
        # Set up pen for crosshair
        pen = QPen(QColor(255, 0, 0))  # Red crosshair
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Draw crosshair
        line_length = 8
        painter.drawLine(center - line_length, center, center + line_length, center)  # Horizontal
        painter.drawLine(center, center - line_length, center, center + line_length)  # Vertical
        
        # Draw circle
        painter.drawEllipse(center - 4, center - 4, 8, 8)

    def _update_center_cursor(self):
        """Update position of center cursor and its coordinate display."""
        # Position the cursor at the center
        center_x = self.width() // 2 - self.center_cursor.width() // 2
        center_y = self.height() // 2 - self.center_cursor.height() // 2
        self.center_cursor.move(center_x, center_y)
        
        # Calculate world coordinates at screen center
        world_x, world_y = self._screen_to_world(self.width() // 2, self.height() // 2)
        
        # Update label text and position
        self.center_coordinates_label.setText(f"X: {world_x:.3f}, Y: {world_y:.3f}")
        label_x = center_x - self.center_coordinates_label.width() // 2 + 10
        label_y = center_y - 30
        self.center_coordinates_label.move(label_x, label_y)


# Demo
# -------------------------------------------------------------------------------------------------


class Map3DDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Visualization Demo")
        self.resize(800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create 3D map widget
        self.map_3d = Map3D()
        main_layout.addWidget(self.map_3d, 3)

        # Create controls widget
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        main_layout.addWidget(controls_widget, 1)

        # Rotation controls
        self.roll_slider = self._create_slider_group("Roll", controls_layout)
        self.pitch_slider = self._create_slider_group("Pitch", controls_layout)
        self.yaw_slider = self._create_slider_group("Yaw", controls_layout)

        # Animation button
        buttons_layout = QHBoxLayout()
        controls_layout.addLayout(buttons_layout)

        self.animate_button = QPushButton("Start Animation")
        self.animate_button.setCheckable(True)
        self.animate_button.clicked.connect(self._toggle_animation)
        buttons_layout.addWidget(self.animate_button)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._reset_orientation)
        buttons_layout.addWidget(reset_button)

        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation)
        self.animation_angle = 0

        # Initial update
        self._update_orientation()

        # Car controls
        car_controls_label = QLabel("Car Controls")
        car_controls_label.setAlignment(Qt.AlignCenter)
        controls_layout.addWidget(car_controls_label)

        car_controls = QHBoxLayout()
        controls_layout.addLayout(car_controls)

        # Car position controls
        car_x_layout = QVBoxLayout()
        car_controls.addLayout(car_x_layout)
        car_x_layout.addWidget(QLabel("X Position"))
        self.car_x_slider = QSlider(Qt.Horizontal)
        self.car_x_slider.setMinimum(-50)
        self.car_x_slider.setMaximum(50)
        self.car_x_slider.setValue(0)
        self.car_x_slider.valueChanged.connect(self._update_car_position)
        car_x_layout.addWidget(self.car_x_slider)

        car_y_layout = QVBoxLayout()
        car_controls.addLayout(car_y_layout)
        car_y_layout.addWidget(QLabel("Y Position"))
        self.car_y_slider = QSlider(Qt.Horizontal)
        self.car_y_slider.setMinimum(-50)
        self.car_y_slider.setMaximum(50)
        self.car_y_slider.setValue(0)
        self.car_y_slider.valueChanged.connect(self._update_car_position)
        car_y_layout.addWidget(self.car_y_slider)

        car_heading_layout = QVBoxLayout()
        car_controls.addLayout(car_heading_layout)
        car_heading_layout.addWidget(QLabel("Heading"))
        self.car_heading_slider = QSlider(Qt.Horizontal)
        self.car_heading_slider.setMinimum(0)
        self.car_heading_slider.setMaximum(359)
        self.car_heading_slider.setValue(0)
        self.car_heading_slider.valueChanged.connect(self._update_car_position)
        car_heading_layout.addWidget(self.car_heading_slider)

    # Add to Map3DDemo class
    def _update_car_position(self):
        """Update car position based on slider values."""
        x = self.car_x_slider.value() / 10.0  # Scale for finer control
        y = self.car_y_slider.value() / 10.0
        heading = self.car_heading_slider.value()
        self.map_3d.move_car(x, y, heading)

    def _create_slider_group(self, name, parent_layout):
        """Create a labeled slider for a rotation axis."""
        layout = QHBoxLayout()
        label = QLabel(f"{name}: 0°")

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(-180)
        slider.setMaximum(180)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(45)

        slider.valueChanged.connect(lambda val: self._slider_changed(val, name, label))

        layout.addWidget(QLabel(name))
        layout.addWidget(slider)
        layout.addWidget(label)

        parent_layout.addLayout(layout)
        return slider

    def _slider_changed(self, value, name, label):
        """Update label and 3D view when slider changes."""
        label.setText(f"{name}: {value}°")
        self._update_orientation()

    def _update_orientation(self):
        """Update the 3D view with current rotation values."""
        rotation = Rotation(
            roll=self.roll_slider.value(),
            pitch=self.pitch_slider.value(),
            yaw=self.yaw_slider.value(),
        )
        self.map_3d.update_data(rotation)

    def _toggle_animation(self, checked):
        """Toggle animation on/off."""
        if checked:
            self.animate_button.setText("Stop Animation")
            self.animation_timer.start(30)
            # Disable sliders during animation
            self.roll_slider.setEnabled(False)
            self.pitch_slider.setEnabled(False)
            self.yaw_slider.setEnabled(False)
        else:
            self.animate_button.setText("Start Animation")
            self.animation_timer.stop()
            # Enable sliders after animation
            self.roll_slider.setEnabled(True)
            self.pitch_slider.setEnabled(True)
            self.yaw_slider.setEnabled(True)

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
        self.roll_slider.setValue(int(rotation.roll))
        self.pitch_slider.setValue(int(rotation.pitch))
        self.yaw_slider.setValue(int(rotation.yaw))

        # Update the 3D view
        self.map_3d.update_data(rotation)

    def _reset_orientation(self):
        """Reset to default orientation."""
        self.roll_slider.setValue(0)
        self.pitch_slider.setValue(0)
        self.yaw_slider.setValue(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Map3DDemo()
    window.show()
    sys.exit(app.exec())
