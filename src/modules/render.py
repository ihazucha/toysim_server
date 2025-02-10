import sys

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import (
    QPixmap,
    QIcon,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QSpacerItem,
    QLabel,
    QWidget,
    QSizePolicy,
    QTabWidget,
)

from modules.ui.plots import (
    EncodersPlotWidget,
    IMUPlotWidget,
    MapPlotWidget,
    SpeedPlotWidget,
    SteeringPlotWidget,
)

from utils.data import icon_path

from modules.ui.sidebar import RecordSidebar
from modules.ui.recorder import RecordingThread
from modules.ui.toolbar import TopToolBar
from modules.ui.config import ConfigPanel
from modules.ui.data import ImageDataThread, SensorDataThread


class RendererMainWindow(QMainWindow):
    init_complete = Signal()

    def __init__(self):
        super().__init__()

    def init(self):
        self._init_main_window()
        self._init_sidebar()
        self._init_tabs()
        self._init_camera_rgb()
        self._init_camera_depth()
        self._init_speed_plot()
        self._init_steering_plot()
        self._init_map_plot()
        self._init_imu_plot()
        self._init_plt_encoders()
        self._init_config_panel()  # Add this line

        self._init_layout()
        self._init_top_toolbar()

        self.showNormal()
        self.init_complete.emit()

    def _init_tabs(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(False)
        self.tab_widget.setMovable(True)
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setFocusPolicy(Qt.StrongFocus)
        self.tab_widget.setFocus()

        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.tab_widget.addTab(self.tab1, "Main Layout")
        self.tab_widget.addTab(self.tab2, "Custom Layout")

        self.setCentralWidget(self.tab_widget)

    def _init_layout(self):
        imu_layout = QVBoxLayout()
        imu_layout.addWidget(self.map_plot, stretch=1)
        imu_layout.addWidget(self.imu_plot, stretch=1)

        encoders_layout = QVBoxLayout()
        encoders_layout.addWidget(self.plt_encoders)
        encoders_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        left_layout = QHBoxLayout()
        left_layout.addLayout(imu_layout)
        # left_layout.addLayout(encoders_layout)

        middle_layout = QVBoxLayout()
        middle_layout.addWidget(self.rgb_label)
        middle_layout.addWidget(self.depth_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.speed_plot)
        right_layout.addWidget(self.steering_plot)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(middle_layout)
        main_layout.addLayout(right_layout)

        self.tab1.setLayout(main_layout)
        self.setStyleSheet(
            """
            background-color: #2d2a2e;
            """
        )

    def _init_main_window(self):
        self.setWindowTitle("ToySim UI")
        self.setWindowIcon(QIcon(icon_path("toysim_icon")))

    def _init_top_toolbar(self):
        self.top_tool_bar = TopToolBar(parent=self.centralWidget())
        self.addToolBar(Qt.TopToolBarArea, self.top_tool_bar)

    def _init_sidebar(self):
        self.record_sidebar = RecordSidebar(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.record_sidebar)

    def _init_camera_rgb(self):
        self.rgb_label = QLabel(self)
        self.rgb_label.setMinimumSize(640, 480)
        self.rgb_pixmap = QPixmap()

    def _init_camera_depth(self):
        self.depth_label = QLabel(self)
        self.depth_label.setMinimumSize(640, 480)
        self.depth_pixmap = QPixmap()

    def _init_speed_plot(self):
        self.speed_plot = SpeedPlotWidget()

    def _init_steering_plot(self):
        self.steering_plot = SteeringPlotWidget()

    def _init_map_plot(self):
        self.map_plot = MapPlotWidget()

    def _init_imu_plot(self):
        self.imu_plot = IMUPlotWidget()

    def _init_plt_encoders(self):
        self.plt_encoders = EncodersPlotWidget()

    def _init_config_panel(self):
        self.config_panel = ConfigPanel(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.config_panel)

    def update_image_data(self, data):
        qimage, timestamp = data
        self.rgb_pixmap.convertFromImage(qimage)
        self.rgb_label.setPixmap(self.rgb_pixmap)
        # Depth
        # self.depth_pixmap.convertFromImage(qimage_depth)
        # self.depth_label.setPixmap(self.depth_pixmap)

    def update_sensor_data(self, data):
        self.plt_encoders.update(data.rleft_encoder, data.rright_encoder)


class Renderer:
    def __init__(self):
        pass

    def run(self):
        app = QApplication(sys.argv)
        window = RendererMainWindow()

        t_image_data = ImageDataThread()
        t_image_data.data_ready.connect(window.update_image_data)
        window.init_complete.connect(t_image_data.start)

        t_sensor_data = SensorDataThread()
        t_sensor_data.data_ready.connect(window.update_sensor_data)
        window.init_complete.connect(t_sensor_data.start)

        t_rec = RecordingThread()
        window.init_complete.connect(t_rec.start)

        threads = [t_image_data, t_sensor_data, t_rec]

        def stop_threads():
            [t.exit() for t in threads]
            [t.wait() for t in threads]

        app.aboutToQuit.connect(stop_threads)

        window.init()
        window.top_tool_bar.record_toggled.connect(t_rec.toggle)

        return app.exec()
