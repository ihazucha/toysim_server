import sys
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSizePolicy,
    QTabWidget,
    QLabel,
)

from modules.ui.plots import (
    EncoderPlotWidget,
    MapPlotWidget,
    SteeringPlotWidget,
    LongitudinalControlWidget,
    LatencyPlotWidget,
)
from modules.ui.presets import Colors
from modules.ui.widgets.map_3d import Map3D

from modules.messaging import messaging
from utils.paths import icon_path

from modules.ui.sidebar import RecordsSidebar
from modules.ui.recorder import RecorderThread, PlaybackThread
from modules.ui.toolbar import TopToolBar
from modules.ui.config import ConfigSidebar
from modules.ui.data import SimDataThread, VehicleDataThread, QSimData, QRealData
from modules.ui.presets import Fonts, FitGraphicsView
from modules.ui.settings import WindowSettings

from collections import deque
import time


# Utils
# -------------------------------------------------------------------------------------------------


def toggle_widget(w: QWidget):
    w.close() if w.isVisible() else w.show()


# UI
# -------------------------------------------------------------------------------------------------


from collections import deque
import time
import cProfile  # Add cProfile
import pstats  # Add pstats
import io  # Add io


class RendererMainWindow(QMainWindow):
    init_complete = Signal()

    def __init__(self):
        super().__init__()

        self.settings = WindowSettings(self)
        self.settings.load()

        # --- FPS Tracking ---
        self._last_paint_time = None
        self._gui_fps_samples = deque([0] * 10, maxlen=10)

        self._last_update_time = None
        self._update_fps_samples = deque([0] * 10, maxlen=20) # Average over 10 paints
        # ---

    def closeEvent(self, event):
        self.settings.save()
        event.accept()

    def init(self):
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: {Colors.PRIMARY};
            }}
        """
        )
        self.setWindowTitle("ToySim UI")
        self.setWindowIcon(QIcon(icon_path("toysim_icon")))

        self._init_tabs()
        self.config_sidebar = self._init_config_sidebar()
        self.records_sidebar = self._init_records_sidebar()
        self.top_tool_bar = self._init_top_toolbar(self.config_sidebar, self.records_sidebar)
        self._init_status_bar()

        self.showNormal()
        self.init_complete.emit()

    def _init_records_sidebar(self):
        records_sidebar = RecordsSidebar(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, records_sidebar)
        return records_sidebar

    def _init_config_sidebar(self):
        config_sidebar = ConfigSidebar(self)
        self.addDockWidget(Qt.RightDockWidgetArea, config_sidebar)
        return config_sidebar

    def _init_top_toolbar(self, config_sidebar, records_sidebar):
        top_tool_bar = TopToolBar(parent=self.centralWidget())
        self.addToolBar(Qt.TopToolBarArea, top_tool_bar)
        top_tool_bar.config_panel_toggled.connect(lambda: toggle_widget(config_sidebar))
        top_tool_bar.records_panel_toggled.connect(lambda: toggle_widget(records_sidebar))
        return top_tool_bar

    def _init_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(False)
        self.tabs.setMovable(False)
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.North)

        tab1 = self._init_tab1_layout()
        tab2 = self._init_tab2_layout()

        self.tabs.addTab(tab1, "Dashboard")
        self.tabs.addTab(tab2, "Stats")

        self.setCentralWidget(self.tabs)

    def _init_tab1_layout(self):
        self.rgb_graphics_view = FitGraphicsView(self)
        self.depth_graphics_view = FitGraphicsView(self)
        self.speed_plot = LongitudinalControlWidget()
        self.steering_plot = SteeringPlotWidget()
        self.map_plot = MapPlotWidget()
        self.map3d_plot = Map3D()

        imu_layout = QVBoxLayout()
        imu_layout.addWidget(self.map_plot, stretch=1)
        imu_layout.addWidget(self.map3d_plot, stretch=1)

        left_layout = QHBoxLayout()
        left_layout.addLayout(imu_layout)

        middle_layout = QVBoxLayout()
        middle_layout.addWidget(self.rgb_graphics_view)
        middle_layout.addWidget(self.depth_graphics_view)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.speed_plot)
        right_layout.addWidget(self.steering_plot)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(9)
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(middle_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=1)

        tab1 = QWidget()
        tab1.setLayout(main_layout)
        return tab1

    def _init_tab2_layout(self):
        self.processor_period_plot = LatencyPlotWidget(name="T Processor", fps_target=30)
        self.processor_dt_plot = LatencyPlotWidget(name="dt Processor", fps_target=30)

        layout = QHBoxLayout()
        layout.setSpacing(9)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.processor_period_plot)
        left_layout.addWidget(self.processor_dt_plot)

        right_layout = QVBoxLayout()
        self.left_encoder_plot = EncoderPlotWidget(name="Left Rear Encoder")
        self.left_encoder_plot.setMinimumSize(0, 0)
        self.left_encoder_plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.right_encoder_plot = EncoderPlotWidget(name="Right Rear Encoder")
        self.right_encoder_plot.setMinimumSize(0, 0)
        self.right_encoder_plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        right_layout.addWidget(self.left_encoder_plot)
        right_layout.addWidget(self.right_encoder_plot)

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        tab2 = QWidget()
        tab2.setLayout(layout)
        return tab2

    def _init_status_bar(self):
        status_bar = self.statusBar()
        status_bar.setStyleSheet(
            f"""
            QStatusBar {{
                background-color: {Colors.ACCENT};
                color: {Colors.ON_ACCENT};
            }}
        """
        )
        status_bar.setSizeGripEnabled(False)

        self.fps_label = QLabel("Data FPS: --") # Rename old label
        self.fps_label.setStyleSheet(f"color: {Colors.ON_FOREGROUND}; padding-right: 5px;")
        status_bar.addPermanentWidget(self.fps_label)

        # Add new label for GUI FPS
        self.gui_fps_label = QLabel("GUI FPS: --")
        self.gui_fps_label.setStyleSheet(f"color: {Colors.ON_FOREGROUND}; padding-right: 5px;")
        status_bar.addPermanentWidget(self.gui_fps_label)

    def paintEvent(self, event):
        """Override paintEvent to measure GUI frame time."""
        # --- GUI FPS Measurement ---
        current_paint_time = time.perf_counter()
        if self._last_paint_time is not None:
            dt = current_paint_time - self._last_paint_time
            if dt > 1e-9:
                fps = 1.0 / dt
                self._gui_fps_samples.append(fps)
                avg_fps = sum(self._gui_fps_samples) / len(self._gui_fps_samples)
                self.gui_fps_label.setText(f"GUI FPS: {avg_fps:.1f}")
        self._last_paint_time = current_paint_time
        # ---
        super().paintEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        super().keyPressEvent(event)

    def update_simulation_data(self, data: QSimData):
        rgb_pixmap = QPixmap.fromImage(data.rgb_qimage)
        self.rgb_graphics_view.set_pixmap(rgb_pixmap)

        depth_pixmap = QPixmap.fromImage(data.depth_qimage)
        self.depth_graphics_view.set_pixmap(depth_pixmap)

        period_ms = data.processor_period_ns / 1e6
        self.processor_period_plot.update(dt_ms=period_ms)
        dt_ms = data.processor_dt_ns / 1e6
        self.processor_dt_plot.update(dt_ms=dt_ms)

    def update_real_data(self, data: QRealData):
        # --- Data Update Rate ---
        current_time = time.time()
        if self._last_update_time is not None:
            dt = current_time - self._last_update_time
            if dt > 0:
                fps = 1.0 / dt
                self._update_fps_samples.append(fps)
                avg_fps = sum(self._update_fps_samples) / len(self._update_fps_samples)
                self.fps_label.setText(f"Data FPS: {avg_fps:.1f}")
        self._last_update_time = current_time
        #  ---

        rgb_pixmap = QPixmap.fromImage(data.rgb_qimage)
        self.rgb_graphics_view.set_pixmap(rgb_pixmap)

        rgb_updated_pixmap = QPixmap.fromImage(data.rgb_updated_qimage)
        self.depth_graphics_view.set_pixmap(rgb_updated_pixmap)

        self.map3d_plot.update_data(
            car_x=0,
            car_y=0,
            car_heading=0,
            car_steering_angle=data.raw.original.control.steering_angle,
            roadmarks=data.raw.roadmarks_data.roadmarks,
            path=data.raw.roadmarks_data.path,
        )
        self.speed_plot.update(
            measured_speed=data.raw.original.sensor_fusion.avg_speed,
            target_speed=data.raw.original.control.speed,
            engine_power=data.raw.original.actuators.motor_power,
        )
        encoder_data_samples = [x.encoder_data for x in data.raw.original.sensor_fusion.speedometer]
        self.left_encoder_plot.update(encoder_data_samples)


class Renderer:
    def run(self, return_window=False):
        app = QApplication.instance() or QApplication(sys.argv)
        app.setFont(Fonts.GUIMonospace)
        window = RendererMainWindow()

        t_sim_data = SimDataThread()
        t_sim_data.data_ready.connect(window.update_simulation_data)
        # window.init_complete.connect(t_sim_data.start)

        t_vehicle_data = VehicleDataThread()
        t_vehicle_data.data_ready.connect(window.update_real_data)
        window.init_complete.connect(t_vehicle_data.start)

        t_recorder = RecorderThread(data_queue=messaging.q_processing)
        window.init_complete.connect(t_recorder.start)

        t_playback = PlaybackThread(data_queue=messaging)
        window.init_complete.connect(t_playback.start)

        threads = [t_sim_data, t_vehicle_data, t_recorder, t_playback]

        def stop_threads():
            print("Shutting down threads gracefully...")

            for t in threads:
                if hasattr(t, "requestInterruption"):
                    t.requestInterruption()

            # Give threads time to process the interruption request
            QApplication.processEvents()

            for t in threads:
                if hasattr(t, "stop"):
                    try:
                        t.stop()
                    except Exception as e:
                        print(f"Error stopping thread {t}: {e}")

            # Wait with timeout
            for t in threads:
                if not t.wait(1000):  # 1 second timeout
                    print(f"Thread {t} did not quit in time, forcing termination")
                    t.terminate()
                    t.wait()

        app.aboutToQuit.connect(stop_threads)

        window.init()
        window.top_tool_bar.record_toggled.connect(t_recorder.toggle)
        window.top_tool_bar.playback_toggled.connect(t_playback.toggle)
        window.records_sidebar.record_selected.connect(t_playback.set_current_record)
        window.records_sidebar.record_selected.connect(window.top_tool_bar.handle_record_selected)

        window.records_sidebar.record_selected.emit("1743006114457116600")
        window.top_tool_bar.playback_toggled.emit(True)

        self.app = app
        self.window = window

        if return_window:
            return window
        else:
            return app.exec()
