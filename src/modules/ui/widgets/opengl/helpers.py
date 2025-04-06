import numpy as np

from PySide6.QtGui import QFont

from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLTextItem
from pyqtgraph import Transform3D, Vector

from modules.ui.presets import MColors
from modules.ui.widgets.opengl.shapes import OpaqueCube


class BasisVectors3D:
    BASE_FONT_SIZE = 16
    MIN_FONT_SIZE = 6
    FONT = QFont("Monospace", BASE_FONT_SIZE)

    def __init__(self, parent_widget: GLViewWidget, name: str, show_xyz_labels=False):
        self.parent_widget = parent_widget
        self.name = name

        len = 0.05
        self.x = GLLinePlotItem(pos=np.array([[0] * 3, [len, 0, 0]]), color=MColors.RED, width=3)
        self.y = GLLinePlotItem(pos=np.array([[0] * 3, [0, len, 0]]), color=MColors.GREEN, width=3)
        self.z = GLLinePlotItem(pos=np.array([[0] * 3, [0, 0, len]]), color=MColors.BLUE, width=3)
        self.xyz_lines = [self.x, self.y, self.z]

        dist = len + len / 5
        self.x_label = GLTextItem(pos=np.array([dist, len / 12, 0]), text=f"X", color=MColors.RED)
        self.y_label = GLTextItem(pos=np.array([len / 12, dist, 0]), text=f"Y", color=MColors.GREEN)
        self.z_label = GLTextItem(pos=np.array([0, len / 12, dist]), text=f"Z", color=MColors.BLUE)
        self.toggle_axes_labels(show_xyz_labels)

        size = len / 8
        self.origin = OpaqueCube(size=size)
        self.origin_offset = Vector(*[-size / 2] * 3)
        self.origin_label = GLTextItem(pos=self.origin_offset, color=MColors.WHITE)
        self._add_to_parent_widget(parent_widget)

    def update_origin_label(self):
        # Use origin of X line as the definite position
        x, y, z = self.x.transform().matrix()[:3, 3]
        self.origin_label.setData(text=f"{self.name}({x:.3f}, {y:.3f}, {z:.3f})")

    def toggle_axes_labels(self, show: bool):
        self.xyz_labels = [self.x_label, self.y_label, self.z_label] if show else []

    def _add_to_parent_widget(self, widget: GLViewWidget):
        for line in self.xyz_lines:
            widget.addItem(line)
        for label in self.xyz_labels:
            widget.addItem(label)

        widget.addItem(self.origin)
        widget.addItem(self.origin_label)

    def scale_font_by_camera_distance(self, distance: float):
        scale = 2.5
        size = self.BASE_FONT_SIZE / (distance * scale)
        size = max(self.MIN_FONT_SIZE, size)
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