from PySide6.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QLabel,
    QSlider,
    QHBoxLayout,
    QLineEdit,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator

from modules.messaging import messaging
from datalink.data import PurePursuitConfig


class FloatSlider(QWidget):
    def __init__(self, key, label_text, min, max, default, step, config_panel, parent=None):
        super().__init__(parent)
        self.key = key
        self.config_panel = config_panel
        self.init_ui(label_text, min, max, default, step)

    def init_ui(self, label_text, min, max, default, step):
        layout = QVBoxLayout()
        label_layout = QHBoxLayout()
        label = QLabel(label_text)
        self.value_input = QLineEdit()
        self.value_input.setReadOnly(False)
        self.value_input.setFixedWidth(70)
        self.value_input.setText(str(default))
        self.value_input.setValidator(QDoubleValidator(min, max, 2))

        label_layout.addWidget(label)
        label_layout.addWidget(self.value_input)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min)
        self.slider.setMaximum(max)
        self.slider.setValue(default)
        self.slider.setSingleStep(step)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(step)
        self.slider.setFixedWidth(200)

        self.slider.valueChanged.connect(self.update_text_input)
        self.value_input.textChanged.connect(self.update_slider_value)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel(f"{min:.2f}"))
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(QLabel(f"{max:.2f}"))

        layout.addLayout(label_layout)
        layout.addLayout(slider_layout)
        self.setLayout(layout)

    def update_text_input(self, value):
        self.config_panel.update_data(self.key, value)
        self.value_input.setText(f"{value:.2f}")

    def update_slider_value(self, text):
        try:
            value = float(text)
            if self.slider.minimum() <= value <= self.slider.maximum():
                self.slider.blockSignals(True)
                self.slider.setValue(value)
                self.slider.blockSignals(False)
        except ValueError:
            pass  # Ignore invalid


class ConfigSidebar(QDockWidget):
    def __init__(self, parent=None, default_closed=True):
        super().__init__("Config", parent)
        self.data = PurePursuitConfig.new_alamak()
        self.init_ui()
        self._q_ui = messaging.q_ui.get_producer()
        
        if default_closed:
            self.close()

    def init_ui(self):
        self.config_widget = QWidget()
        self.setWidget(self.config_widget)

        self.lookahead_factor = FloatSlider(
            "lookahead_factor",
            "Lookahead Factor",
            0,
            10,
            self.data.lookahead_factor,
            0.05,
            self,
        )
        self.lookahead_l_min = FloatSlider(
            "lookahead_dist_min",
            "Lookahead Min Distance",
            0,
            10,
            self.data.lookahead_dist_min,
            0.1,
            self,
        )
        self.lookahead_l_max = FloatSlider(
            "lookahead_dist_max",
            "Lookahead Max Dist",
            0,
            50,
            self.data.lookahead_dist_max,
            0.1,
            self,
        )

        layout = QVBoxLayout()
        layout.addWidget(self.lookahead_factor)
        layout.addWidget(self.lookahead_l_min)
        layout.addWidget(self.lookahead_l_max)

        layout.addStretch()
        self.config_widget.setLayout(layout)

    def update_data(self, key, value):
        # TODO: think of a more reliable messaging architecture, and closer to callbacks
        setattr(self.data, key, value)
        self._q_ui.put(self.data)
