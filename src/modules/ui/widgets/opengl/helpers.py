import numpy as np

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QVector3D, QFont
from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLTextItem
from pyqtgraph import Transform3D, Vector

from modules.ui.presets import GLColors, Fonts
from modules.ui.widgets.opengl.shapes import OpaqueCube

# Functions
# -------------------------------------------------------------------------------------------------


def get_camera_forward_vector(elevation, azimuth):
    elevation_rad = np.radians(elevation)
    azimuth_rad = np.radians(azimuth)
    x = -np.cos(azimuth_rad) * np.cos(elevation_rad)
    y = -np.sin(azimuth_rad) * np.cos(elevation_rad)
    z = -np.sin(elevation_rad)
    return Vector(x, y, z)


def get_camera_right_vector(azimuth):
    azimuth_rad = np.radians(azimuth)
    x = -np.sin(azimuth_rad)
    y = np.cos(azimuth_rad)
    z = 0
    return Vector(x, y, z)


def get_camera_ground_intersection(cam_position: Vector, cam_direction: Vector):
    if cam_position.z() < 0:
        # No ground is seen
        return None

    distance = -cam_position.z() / cam_direction.z()
    intersection_x = cam_position.x() + distance * cam_direction.x()
    intersection_y = cam_position.y() + distance * cam_direction.y()

    return Vector(intersection_x, intersection_y, 0)


# Classes
# -------------------------------------------------------------------------------------------------


class ReferenceFrame:
    def __init__(
        self,
        parent_widget: GLViewWidget,
        name: str,
        size=0.025,
        show_cube=True,
    ):
        self.parent_widget = parent_widget
        self.name = name
        self.size = size
        self.show_cube = show_cube

        self._is_focused = False
        self._scale_factor = 1

        self._default_color = GLColors.WHITE
        self._default_font = QFont(Fonts.OpenGLMonospace)
        self._default_font.setBold(True)

        x_line = np.array([[0] * 3, [size, 0, 0]])
        self.x = GLLinePlotItem(pos=x_line, color=GLColors.RED, width=3, antialias=True)
        y_line = np.array([[0] * 3, [0, size, 0]])
        self.y = GLLinePlotItem(pos=y_line, color=GLColors.GREEN, width=3, antialias=True)
        z_line = np.array([[0] * 3, [0, 0, size]])
        self.z = GLLinePlotItem(pos=z_line, color=GLColors.BLUE, width=3, antialias=True)
        self.xyz_lines = [self.x, self.y, self.z]

        radius = size / 8
        self.origin_offset = Vector(*[-radius / 2] * 3)
        self.origin = OpaqueCube(size=radius)
        self.origin.setDepthValue(-1)
        self.label_offset = Vector(-0.005, -0.005, 0.01)
        self.origin_label = GLTextItem(
            pos=self.origin_offset, color=self._default_color, font=self._default_font
        )

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

        if self._is_focused:
            tr.translate(self.label_offset)

        self.origin_label.setTransform(tr)
        self._update_origin_label()

    def set_focus(self, focused: bool):
        if self._is_focused == focused:
            return
        self._is_focused = focused

        # Calculate new size based on active state
        current_size = self.size * (1.5 if focused else 1.0)

        # Update axes with new size
        for line, direction in zip(self.xyz_lines, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            end_point = [d * current_size for d in direction]
            line.setData(pos=np.array([[0, 0, 0], end_point]))

        if self._is_focused:
            font = QFont(self._default_font)
            self.origin_label.setData(color=GLColors.ORANGE, font=font)
            self.scale_factor = 2
        else:
            self.scale_factor = 1
            self.origin_label.setData(color=self._default_color, font=self._default_font)

        self.setTransform(self.get_transform())

    def is_focused(self) -> bool:
        return self._is_focused

    def _update_origin_label(self):
        self.origin_label.setData(text=f"{self.name}")

    def _add_to_parent_widget(self, widget: GLViewWidget):
        for line in self.xyz_lines:
            widget.addItem(line)

        if self.show_cube:
            widget.addItem(self.origin)
        widget.addItem(self.origin_label)


class GLMultiTextItem(GLTextItem):
    def __init__(self, parentItem=None, colors: list = [], texts: list = [], **kwds):
        super().__init__(parentItem, **kwds)
        self.colors = colors if len(colors) else self.color
        self.texts = texts

    def setData(self, colors: list = [], texts: list = [], **kwds):
        assert len(colors) <= 1 or len(colors) == len(
            texts
        ), "Number of colors has to match texts or be <= 1"

        if len(colors):
            self.colors = colors
        self.texts = texts
        super().setData(**kwds)
        self.update()

    def paint(self):
        if len(self.texts) < 1:
            return
        self.setupGLState()

        project = self.compute_projection()
        vec3 = QVector3D(*self.pos)
        text_pos = project.map(vec3).toPointF()

        painter = QPainter(self.view())
        painter.setFont(self.font)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
        )

        for i, text in enumerate(self.texts):
            color = self.colors[i] if len(self.colors) > 1 else self.colors[0]
            painter.setPen(color)
            painter.drawText(text_pos + QPointF(0, -i * 14), text)

        painter.end()
