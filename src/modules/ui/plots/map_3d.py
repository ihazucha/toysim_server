from copy import deepcopy
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QVector3D, QFont

from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLGridItem, GLTextItem
from pyqtgraph import Transform3D

import sys
import numpy as np
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
from PySide6.QtCore import Qt, QTimer

sys.path.append("C:/Users/ihazu/Desktop/projects/toysim_server/src")
from modules.ui.plots import Colors
from datalink.data import Rotation

from pyqtgraph.opengl import GLMeshItem, GLBoxItem, MeshData

# Add this function outside any class
def create_cylinder_mesh(radius=1.0, height=1.0, segments=20):
    """Create a cylinder mesh with the given dimensions."""
    # Create points for top and bottom circles
    theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
    
    # Top circle points (at z = height/2)
    x_top = radius * np.cos(theta)
    y_top = radius * np.sin(theta)
    z_top = np.ones_like(theta) * height/2
    
    # Bottom circle points (at z = -height/2)
    x_bottom = radius * np.cos(theta)
    y_bottom = radius * np.sin(theta)
    z_bottom = np.ones_like(theta) * -height/2
    
    # Combine all points
    vertices = []
    for i in range(segments):
        # Top circle center
        vertices.append([0, 0, height/2])
        # Top circle point i
        vertices.append([x_top[i], y_top[i], z_top[i]])
        # Top circle point i+1
        vertices.append([x_top[(i+1)%segments], y_top[(i+1)%segments], z_top[(i+1)%segments]])
        
        # Bottom circle center
        vertices.append([0, 0, -height/2])
        # Bottom circle point i+1
        vertices.append([x_bottom[(i+1)%segments], y_bottom[(i+1)%segments], z_bottom[(i+1)%segments]])
        # Bottom circle point i
        vertices.append([x_bottom[i], y_bottom[i], z_bottom[i]])
        
        # Side quad (two triangles) 
        # Triangle 1
        vertices.append([x_top[i], y_top[i], z_top[i]])
        vertices.append([x_bottom[i], y_bottom[i], z_bottom[i]])
        vertices.append([x_bottom[(i+1)%segments], y_bottom[(i+1)%segments], z_bottom[(i+1)%segments]])
        
        # Triangle 2
        vertices.append([x_top[i], y_top[i], z_top[i]])
        vertices.append([x_bottom[(i+1)%segments], y_bottom[(i+1)%segments], z_bottom[(i+1)%segments]])
        vertices.append([x_top[(i+1)%segments], y_top[(i+1)%segments], z_top[(i+1)%segments]])
    
    # Convert to numpy array
    vertices = np.array(vertices, dtype=np.float32)
    
    # Create colors (one color per vertex)
    colors = np.ones((len(vertices), 4), dtype=np.float32) * 0.5  # Gray
    colors[:, 3] = 1.0  # Alpha = 1.0
    
    # Create mesh data
    md = MeshData(vertexes=vertices, vertexColors=colors)
    return md

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
    def __init__(self, parent_widget: GLViewWidget):
        self.x = GLLinePlotItem(pos=np.array([[0] * 3, [0.05, 0, 0]]), color=(1, 0, 0, 1), width=3)
        self.y = GLLinePlotItem(pos=np.array([[0] * 3, [0, 0.05, 0]]), color=(0, 1, 0, 1), width=3)
        self.z = GLLinePlotItem(pos=np.array([[0] * 3, [0, 0, 0.05]]), color=(0, 0, 1, 1), width=3)

        # Create axis labels with improved settings
        self.base_font_size = 16
        self.x_label = GLTextItem(pos=np.array([0.06, 0, 0]), text="X(w)", color=(255, 0, 0, 255))
        self.y_label = GLTextItem(pos=np.array([0, 0.06, 0]), text="Y(w)", color=(0, 255, 0, 255))
        self.z_label = GLTextItem(pos=np.array([0, 0, 0.06]), text="Z(w)", color=(0, 0, 255, 255))

        if parent_widget:
            self.add_to_widget(parent_widget)

    def update_font_scaling(self, distance):
        scale_factor = 2.5
        font_size = self.base_font_size / (distance * scale_factor)
        font_size = max(6, font_size)
        try:
            for label in [self.x_label, self.y_label, self.z_label]:
                label.setData(font=QFont("Consolas, Helvetica", int(font_size)))
        except:
            pass

    def add_to_widget(self, widget: GLViewWidget):
        widget.addItem(self.x)
        widget.addItem(self.y)
        widget.addItem(self.z)

        # Add labels
        widget.addItem(self.x_label)
        widget.addItem(self.y_label)
        widget.addItem(self.z_label)

    def transform(self, tr: Transform3D):
        self.x.setTransform(tr)
        self.y.setTransform(tr)
        self.z.setTransform(tr)

        self.x_label.setTransform(tr)
        self.y_label.setTransform(tr)
        self.z_label.setTransform(tr)


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
        self.world_basis_vectors = BasisVectors3D(parent_widget=self)
        self.world_basis_vectors.update_font_scaling(distance=Map3D.INIT_DIST)
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
            # Enable mouse tracking to get coordinates without clicking
            self.setMouseTracking(True)
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._restore_camera_state()
            # Hide coordinate display
            self.coordinate_label.hide()
            self.setCursor(Qt.CursorShape.ArrowCursor)

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
        else:
            return super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev) -> None:
        super().mousePressEvent(ev)
        # Initialize for top-down view mouseMoveEvent
        self._last_mouse_position = ev.position()

    def wheelEvent(self, ev):
        """Override wheel event to update text scaling after zoom."""
        super().wheelEvent(ev)
        if self._top_down_view_enabled and ev.position() is not None:
            self._update_coordinate_display(ev)
        if hasattr(self, "world_basis_vectors"):
            self.world_basis_vectors.update_font_scaling(self.opts["distance"])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_T:
            self.toggle_top_down_view()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _update_coordinate_display(self, ev):
        """Calculate and display world coordinates from mouse position."""
        # Get screen position
        pos = ev.position().toPoint()

        # Calculate world coordinates
        world_x, world_y = self._screen_to_world(pos.x(), pos.y())

        # Update label text
        self.coordinate_label.setText(f"X: {world_x:.2f}, Y: {world_y:.2f}")

        # Position the label near the cursor but ensure it's visible
        label_x = pos.x() + 15
        label_y = pos.y() - 30

        # Ensure label stays within widget bounds
        if label_x + self.coordinate_label.width() > self.width():
            label_x = pos.x() - self.coordinate_label.width() - 5

        if label_y < 0:
            label_y = pos.y() + 15

        self.coordinate_label.move(label_x, label_y)
        self.coordinate_label.show()

    def _screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates in top-down view."""
        width, height = self.width(), self.height()
        distance = self.opts["distance"]

        # Get the center of the map (assumes the center is at (0, 0) in world coordinates)
        center = self.opts["center"]
        center_x, center_y = center.x(), center.y()

        # Calculate the world-space units per screen pixel
        # This depends on the camera distance (zoom level)
        scale_factor = distance * (0.1 / 0.3136)  # Adjust this based on your scene scale

        # Calculate normalized screen coordinates (-1 to 1)
        norm_x = (2.0 * screen_x / width) - 1.0
        norm_y = 1.0 - (2.0 * screen_y / height)  # Flip Y axis

        # Scale to world coordinates
        view_width = width / height * scale_factor
        view_height = scale_factor

        # Calculate world coordinates
        world_x = (norm_y * view_height) + center_x
        world_y = (norm_x * view_width) - center_y

        return world_x, -world_y

    def update_data(self, rotation: Rotation):
        roll, pitch, yaw = (
            np.deg2rad(rotation.roll),
            np.deg2rad(rotation.pitch),
            np.deg2rad(rotation.yaw),
        )

        transform = Transform3D()
        transform.rotate(np.rad2deg(roll), 1, 0, 0)  # Rotate around X-axis
        transform.rotate(np.rad2deg(pitch), 0, 1, 0)  # Rotate around Y-axis
        transform.rotate(np.rad2deg(yaw), 0, 0, 1)  # Rotate around Z-axis

        self.world_basis_vectors.transform(transform)


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
