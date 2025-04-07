import numpy as np

from PySide6.QtGui import QFont

from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLTextItem
from pyqtgraph import Transform3D, Vector

from modules.ui.presets import MColors
from modules.ui.widgets.opengl.shapes import OpaqueCube


class BasisVectors3D:
    FONT = QFont("Monospace", 8)

    def __init__(self, parent_widget: GLViewWidget, name: str, size=0.05):
        self.parent_widget = parent_widget
        self.name = name

        self.x = GLLinePlotItem(pos=np.array([[0] * 3, [size, 0, 0]]), color=MColors.RED, width=3)
        self.y = GLLinePlotItem(pos=np.array([[0] * 3, [0, size, 0]]), color=MColors.GREEN, width=3)
        self.z = GLLinePlotItem(pos=np.array([[0] * 3, [0, 0, size]]), color=MColors.BLUE, width=3)
        self.xyz_lines = [self.x, self.y, self.z]

        radius = size / 8
        self.origin = OpaqueCube(size=radius)
        self.origin_offset = Vector(*[-radius / 2] * 3)
        self.origin_label = GLTextItem(pos=self.origin_offset, color=MColors.WHITE, font=self.FONT)

        self._add_to_parent_widget(parent_widget)
        self.setTransform(Transform3D())

    def get_position(self):
        return Vector(self.x.transform().matrix()[:3, 3])

    def setTransform(self, tr: Transform3D):
        for line in self.xyz_lines:
            line.setTransform(tr)

        tr.translate(self.origin_offset)
        self.origin_label.setTransform(tr)
        self.origin.setTransform(tr)

        self._update_origin_label()

    def _update_origin_label(self):
        p = self.get_position()
        self.origin_label.setData(text=f"{self.name} ({p.x():.3f}, {p.y():.3f}, {p.z():.3f})")

    def _add_to_parent_widget(self, widget: GLViewWidget):
        for line in self.xyz_lines:
            widget.addItem(line)

        widget.addItem(self.origin)
        widget.addItem(self.origin_label)

