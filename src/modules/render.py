import sys
import json
from pathlib import Path
from PySide6.QtCore import Signal, Qt, QRect
from PySide6.QtGui import QGuiApplication, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QSpacerItem,
    QWidget,
    QSizePolicy,
    QTabWidget,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
)

from modules.ui.plots import (
    EncodersPlotWidget,
    IMUPlotWidget,
    MapPlotWidget,
    SpeedPlotWidget,
    SteeringPlotWidget,
)
from modules.ui.plots import LatencyPlotWidget

from utils.paths import icon_path

from modules.ui.sidebar import RecordSidebar
from modules.ui.recorder import RecordingThread
from modules.ui.toolbar import TopToolBar
from modules.ui.config import ConfigPanel
from modules.ui.data import SimDataThread, VehicleDataThread, QSimData, QVehicleData


class FitGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_pixmap(self, pixmap):
        """Set the pixmap and fit it to the view."""
        self.pixmap_item.setPixmap(pixmap)
        if not pixmap.isNull():
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        """Scale the pixmap when the view is resized."""
        if not self.pixmap_item.pixmap().isNull():
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        super().resizeEvent(event)


class RendererMainWindow(QMainWindow):
    init_complete = Signal()

    def __init__(self):
        super().__init__()
        self.settings_path = Path(__file__).parent / "ui/settings/window_settings.json"
        self.load_window_position()

    def closeEvent(self, event):
        self.save_window_position()
        event.accept()

    def load_window_position(self):
        if self.settings_path.exists():
            with open(self.settings_path, "r") as f:
                pos = json.load(f)
                screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
                # If off-screen (e.g. due to monitor change), use default
                if not screen_geometry.contains(
                    QRect(pos["x"], pos["y"], self.width(), self.height())
                ):
                    pos["x"], pos["y"] = 100, 100
                self.move(pos["x"], pos["y"])
                if "width" in pos and "height" in pos:
                    self.resize(pos["width"], pos["height"])
                if pos.get("isMaximized", False):
                    self.showMaximized()

    def save_window_position(self):
        pos = {"x": self.x(), "y": self.y()}
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        pos["width"] = self.width()
        pos["height"] = self.height()
        pos["isMaximized"] = self.isMaximized()
        with open(self.settings_path, "w") as f:
            json.dump(pos, f, indent=4)

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
        self._init_config_panel()

        self._init_layout()
        self._init_top_toolbar()

        self.showNormal()
        self.init_complete.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        super().keyPressEvent(event)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def toggle_config_panel(self):
        if self.config_panel.isVisible():
            print("close")
            self.config_panel.close()
        else:
            self.config_panel.show()
            print("open")

    def toggle_sidebar(self):
        if self.record_sidebar.isVisible():
            self.record_sidebar.close()
        else:
            self.record_sidebar.show()

    def _init_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(False)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setFocusPolicy(Qt.StrongFocus)
        self.tabs.setFocus()

        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.tabs.addTab(self.tab1, "Main Layout")
        self.tabs.addTab(self.tab2, "Custom Layout")

        self.setCentralWidget(self.tabs)

    def _init_layout(self):
        self._init_tab1_layout()
        self._init_tab2_layout()
        self.setStyleSheet(
            """
            background-color: #2d2a2e;
            """
        )

    def _init_tab1_layout(self):
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
        middle_layout.addWidget(self.rgb_graphics_view, stretch=1)
        middle_layout.addWidget(self.depth_graphics_view, stretch=1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.speed_plot, stretch=1)
        right_layout.addWidget(self.steering_plot, stretch=1)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(middle_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=1)

        self.tab1.setLayout(main_layout)
    
    def _init_tab2_layout(self):
        self.processor_period_plot = LatencyPlotWidget(name="T Processor", fps_target=30)
        self.processor_dt_plot = LatencyPlotWidget(name="dt Processor", fps_target=30)
        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.processor_period_plot)
        left_layout.addWidget(self.processor_dt_plot)
        layout.addLayout(left_layout)
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.tab2.setLayout(layout)

    def _init_main_window(self):
        self.setWindowTitle("ToySim UI")
        self.setWindowIcon(QIcon(icon_path("toysim_icon")))

    def _init_top_toolbar(self):
        self.top_tool_bar = TopToolBar(parent=self.centralWidget())
        self.addToolBar(Qt.TopToolBarArea, self.top_tool_bar)
        self.top_tool_bar.config_toggled.connect(self.toggle_config_panel)
        self.top_tool_bar.sidebar_toggled.connect(self.toggle_sidebar)

    def _init_sidebar(self):
        self.record_sidebar = RecordSidebar(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.record_sidebar)
        self.record_sidebar.close()

    def _init_camera_rgb(self):
        self.rgb_graphics_view = FitGraphicsView(self)
        self.rgb_graphics_view.setMinimumSize(0, 0)

    def _init_camera_depth(self):
        self.depth_graphics_view = FitGraphicsView(self)
        self.depth_graphics_view.setMinimumSize(0, 0)

    def _init_speed_plot(self):
        self.speed_plot = SpeedPlotWidget()
        self.speed_plot.setMinimumSize(0, 0)
        self.speed_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _init_steering_plot(self):
        self.steering_plot = SteeringPlotWidget()
        self.steering_plot.setMinimumSize(0, 0)
        self.steering_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _init_map_plot(self):
        self.map_plot = MapPlotWidget()
        self.map_plot.setMinimumSize(0, 0)
        self.map_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _init_imu_plot(self):
        self.imu_plot = IMUPlotWidget()
        self.imu_plot.setMinimumSize(0, 0)
        self.imu_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _init_plt_encoders(self):
        self.plt_encoders = EncodersPlotWidget()
        self.plt_encoders.setMinimumSize(0, 0)
        self.plt_encoders.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _init_config_panel(self):
        self.config_panel = ConfigPanel(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.config_panel)
        self.config_panel.close()

    def update_simulation_data(self, data: QSimData):
        rgb_pixmap = QPixmap.fromImage(data.rgb_qimage)
        self.rgb_graphics_view.set_pixmap(rgb_pixmap)

        depth_pixmap = QPixmap.fromImage(data.depth_qimage)
        self.depth_graphics_view.set_pixmap(depth_pixmap)

        period_ms = data.processor_period_ns / 1e6
        self.processor_period_plot.update(dt_ms=period_ms)
        dt_ms = data.processor_dt_ns / 1e6
        self.processor_dt_plot.update(dt_ms=dt_ms)
        
    def update_vehicle_data(self, data: QVehicleData):
        rgb_pixmap = QPixmap.fromImage(data.rgb_qimage)
        self.rgb_graphics_view.set_pixmap(rgb_pixmap)
        print("Updating vehicle data")

    def update_sensor_data(self, data):
        self.plt_encoders.update(data.rleft_encoder, data.rright_encoder)


class Renderer:
    def run(self):
        app = QApplication(sys.argv)
        window = RendererMainWindow()

        t_sim_data = SimDataThread()
        t_sim_data.data_ready.connect(window.update_simulation_data)
        window.init_complete.connect(t_sim_data.start)

        t_vehicle_data = VehicleDataThread()
        t_vehicle_data.data_ready.connect(window.update_vehicle_data)
        window.init_complete.connect(t_vehicle_data.start)

        t_rec = RecordingThread()
        window.init_complete.connect(t_rec.start)

        threads = [t_sim_data, t_vehicle_data, t_rec]

        def stop_threads():
            # TODO: try to exit gracefully instead of terminate
            [t.terminate() for t in threads]
            [t.wait() for t in threads]

        app.aboutToQuit.connect(stop_threads)

        window.init()
        window.top_tool_bar.record_toggled.connect(t_rec.toggle)

        return app.exec()
