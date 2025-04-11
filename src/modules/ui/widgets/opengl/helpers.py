from tkinter.font import Font
import numpy as np

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QVector3D, QFont
from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLTextItem
from pyqtgraph import Transform3D, Vector

from modules.ui.presets import MColors, Fonts
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
    def __init__(self, parent_widget: GLViewWidget, name: str, abbrev: str = None, size=0.05):
        assert len(name) > 0, "Come on, you can figure out a 1-letter name"
        self.parent_widget = parent_widget
        self.name = name
        self.abbrev = abbrev if abbrev else name[0]

        self._is_active = False
        self._default_color = MColors.WHITE
        self._default_font = Fonts.Monospace

        self.x = GLLinePlotItem(
            pos=np.array([[0] * 3, [size, 0, 0]]), color=MColors.RED, width=3, antialias=True
        )
        self.y = GLLinePlotItem(
            pos=np.array([[0] * 3, [0, size, 0]]), color=MColors.GREEN, width=3, antialias=True
        )
        self.z = GLLinePlotItem(
            pos=np.array([[0] * 3, [0, 0, size]]), color=MColors.BLUE, width=3, antialias=True
        )
        self.xyz_lines = [self.x, self.y, self.z]

        radius = size / 8
        self.origin = OpaqueCube(size=radius)
        self.origin_offset = Vector(*[-radius / 2] * 3)
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
        self.origin_label.setTransform(tr)

        self._update_origin_label()

    def set_focus(self, active: bool):
        self._is_active = active
        if self._is_active:
            font = QFont(Fonts.Monospace)
            font.setBold(True)
            self.origin_label.setData(color=MColors.ORANGE, font=font)
        else:
            self.origin_label.setData(color=self._default_color, font=self._default_font)

    def _update_origin_label(self):
        p = self.get_position()
        self.origin_label.setData(text=f"{self.abbrev} ({p.x():.3f}, {p.y():.3f}, {p.z():.3f})")

    def _add_to_parent_widget(self, widget: GLViewWidget):
        for line in self.xyz_lines:
            widget.addItem(line)

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
