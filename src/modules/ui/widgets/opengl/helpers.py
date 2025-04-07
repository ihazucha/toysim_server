import numpy as np

from PySide6.QtGui import QFont

from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLTextItem
from pyqtgraph import Transform3D, Vector

from modules.ui.presets import MColors
from modules.ui.widgets.opengl.shapes import OpaqueCube


class BasisVectors3D:
    BASE_FONT_SIZE = 12
    MIN_FONT_SIZE = 8
    FONT = QFont("Monospace", BASE_FONT_SIZE)

    def __init__(self, parent_widget: GLViewWidget, name: str, size=0.05, show_xyz_labels=False):
        self.parent_widget = parent_widget
        self.name = name

        self.x = GLLinePlotItem(pos=np.array([[0] * 3, [size, 0, 0]]), color=MColors.RED, width=3)
        self.y = GLLinePlotItem(pos=np.array([[0] * 3, [0, size, 0]]), color=MColors.GREEN, width=3)
        self.z = GLLinePlotItem(pos=np.array([[0] * 3, [0, 0, size]]), color=MColors.BLUE, width=3)
        self.xyz_lines = [self.x, self.y, self.z]

        dist = size + size / 5
        self.x_label = GLTextItem(pos=np.array([dist, size / 12, 0]), text=f"X", color=MColors.RED)
        self.y_label = GLTextItem(pos=np.array([size / 12, dist, 0]), text=f"Y", color=MColors.GREEN)
        self.z_label = GLTextItem(pos=np.array([0, size / 12, dist]), text=f"Z", color=MColors.BLUE)
        self.toggle_axes_labels(show_xyz_labels)

        radius = size / 8
        self.origin = OpaqueCube(size=radius)
        self.origin_offset = Vector(*[-radius / 2] * 3)
        self.origin_label = GLTextItem(pos=self.origin_offset, color=MColors.WHITE)
        
        self._add_to_parent_widget(parent_widget)
        
        self.transform(Transform3D())

    def get_position(self):
        return Vector(self.x.transform().matrix()[:3, 3])

    def update_origin_label(self):
        # Use origin of X line as the definite position
        p = self.get_position()
        self.origin_label.setData(text=f"{self.name}({p.x():.3f}, {p.y():.3f}, {p.z():.3f})")

    def toggle_axes_labels(self, show: bool):
        self.xyz_labels = [self.x_label, self.y_label, self.z_label] if show else []

    def _add_to_parent_widget(self, widget: GLViewWidget):
        for line in self.xyz_lines:
            widget.addItem(line)
        for label in self.xyz_labels:
            widget.addItem(label)

        widget.addItem(self.origin)
        widget.addItem(self.origin_label)

    def on_camera_change(self, opts: dict):
        self.scale_font_by_camera_position(distance=opts["distance"], center=opts["center"])

    def scale_font_by_camera_position(self, distance: float, center: Vector):
        difference = self.get_position() - center
        length = difference.lengthSquared()
        if length > 0:
            size = self.BASE_FONT_SIZE / (distance * length * 2.5)
            size = np.clip(size, self.MIN_FONT_SIZE, self.BASE_FONT_SIZE)
        else:
            size = self.BASE_FONT_SIZE
        self.FONT.setPointSizeF(size)

        for label in self.xyz_labels:
            label.setData(font=self.FONT)

        self.origin_label.setData(font=self.FONT)

    def transform(self, tr: Transform3D):
        for line in self.xyz_lines:
            line.setTransform(tr)
        for label in self.xyz_labels:
            label.setTransform(tr)

        tr.translate(self.origin_offset)
        self.origin_label.setTransform(tr)
        self.origin.setTransform(tr)

        self.update_origin_label()