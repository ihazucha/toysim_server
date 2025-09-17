import sys

from PySide6 import QtGui
from PySide6.QtCore import Signal, Slot, Qt, QTimer
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QWidget,
    QTabWidget,
    QLabel,
    QGroupBox,
)

from modules.messaging import messaging

from utils.env import pdebug
from datalink.data import ControlData
from time import time_ns

from modules.ui.plots import LatencyPlotWidget
from modules.ui.widgets.long_control import LongitudinalControlWidget
from modules.ui.widgets.lat_control import LateralControlWidget
from modules.ui.widgets.encoder import EncoderWidget
from modules.ui.widgets.playback import PlaybackWidget
from modules.ui.widgets.map3d import Map3D
from modules.ui.widgets.imu3d import IMU3D
from modules.ui.widgets.imu import IMURawWidget
from modules.ui.settings import WindowSettings
from modules.ui.presets import (
    UIColors,
    QSSDebug,
    Fonts,
    FitGraphicsView,
    EMALatencyLabel,
    toggle_widget,
    APP_STYLE,
)
from modules.ui.records_bar import RecordsSidebar
from modules.ui.config_bar import ConfigSidebar
from modules.ui.toolbar import TopToolBar
from modules.ui.recorder import RecorderThread, PlaybackThread
from modules.ui.data import RealDataThread, QSimData, QRealData, SimDataThread

# Global config
# -------------------------------------------------------------------------------------------------

# Disable VSync (significant FPS boost)
sfmt = QtGui.QSurfaceFormat()
sfmt.setSwapInterval(0)
QtGui.QSurfaceFormat.setDefaultFormat(sfmt)

# UI
# -------------------------------------------------------------------------------------------------


class UIController:
    def __init__(self):
        self.speed = 0.0
        self.steering_deg = 0.0
        self.is_controller_used = True

        self.control_timer = QTimer()
        self.control_timer.timeout.connect(self.update)
        self.control_timer.start(10)

        self.q_ui = messaging.q_ui.get_producer()

    def on_W(self):
        self.is_controller_used = True
        self.speed = 0 if self.speed < 0 else 1

    def on_S(self):
        self.is_controller_used = True
        self.speed = 0 if self.speed > 0 else -1

    def on_A(self):
        self.is_controller_used = True
        self.steering_deg = max(self.steering_deg - 1, -20)

    def on_D(self):
        self.is_controller_used = True
        self.steering_deg = min(self.steering_deg + 1, 20)

    def update(self):
        if self.is_controller_used:
            print(f"is_controller_used: {self.is_controller_used} | speed: {self.speed} | steering_deg: {self.steering_deg}")
            data = ControlData(timestamp=time_ns(), speed=self.speed, steering_angle=self.steering_deg)
            self.q_ui.put(data)
        else: 
            self.speed = 0
        self.is_controller_used = False

class RendererMainWindow(QMainWindow):
    tab_changed = Signal(int, str)

    def __init__(self):
        super().__init__()

        self.qssdebug = QSSDebug(widget=QApplication.instance())

        self.settings = WindowSettings(self)
        self.settings.load()

        self.ui_controller = UIController()

    def closeEvent(self, event):
        self.settings.save()
        event.accept()

    def init(self):
        # Layout
        self.central_widget = QWidget()
        self.central_widget.setContentsMargins(0, 0, 0, 0)
        central_layout = QVBoxLayout(self.central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        self.setCentralWidget(self.central_widget)
        self.setWindowTitle("ToySim UI")
        self.setWindowIcon(QIcon("icons:toysim_icon.png"))

        # Widgets
        self.tabs = self._init_tabs()
        self.config_sidebar = self._init_config_sidebar()
        self.records_sidebar = self._init_records_sidebar()
        self.top_tool_bar = self._init_top_toolbar(self.config_sidebar, self.records_sidebar)
        self.playback_widget = self._init_playback_bar()
        self._init_status_bar()

        central_layout.addWidget(self.tabs)
        central_layout.addWidget(self.playback_widget)

        # Signals
        self.records_sidebar.record_selected.connect(
            lambda: self.playback_widget.show() if self.playback_widget.isHidden() else None
        )
        self.showNormal()

    def _init_records_sidebar(self):
        rsb = RecordsSidebar(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, rsb)
        rsb.hide()
        return rsb

    def _init_config_sidebar(self):
        csb = ConfigSidebar(self)
        self.addDockWidget(Qt.RightDockWidgetArea, csb)
        return csb

    def _init_playback_bar(self):
        playback_widget = PlaybackWidget()
        playback_widget.hide()
        return playback_widget

    def _init_top_toolbar(self, config_sidebar, records_sidebar):
        ttb = TopToolBar(parent=self.centralWidget())
        self.addToolBar(Qt.TopToolBarArea, ttb)
        ttb.control_panel_toggled.connect(lambda: toggle_widget(config_sidebar))
        ttb.records_panel_toggled.connect(lambda: toggle_widget(records_sidebar))
        return ttb

    def _init_tabs(self):
        tabs = QTabWidget()
        tabs.setTabsClosable(False)
        tabs.setMovable(False)
        tabs.setDocumentMode(True)
        tabs.setTabPosition(QTabWidget.North)
        tabs.setStyleSheet(
            """
            QTabWidget::tab-bar {
                left: 8px; /* move to the right by 5px */
            }
            """
        )

        tab_dashboard = self._init_tab_dashboard_layout()
        tabs.addTab(tab_dashboard, "Dashboard")

        # Lazy loading of the sensors tab
        self.sensors_tab_placeholder = QWidget()
        tabs.addTab(self.sensors_tab_placeholder, "Sensors")

        tab_system = self._init_tab_system_layout()
        tabs.addTab(tab_system, "System")

        tabs.currentChanged.connect(self._handle_tab_changed)

        return tabs

    def _handle_tab_changed(self, index):
        tab_text = self.tabs.tabText(index)

        # Sensors tab will be initiated on first use
        if (
            tab_text == "Sensors"
            and isinstance(self.tabs.widget(index), QWidget)
            and not hasattr(self, "sensors_tab_initialized")
        ):
            self.sensors_tab_initialized = True
            real_sensors_tab = self._init_tab_sensors_layout()
            self.tabs.blockSignals(True)
            self.tabs.removeTab(index)
            self.tabs.insertTab(index, real_sensors_tab, "Sensors")
            self.tabs.setCurrentIndex(index)
            self.tabs.blockSignals(False)

        self.tab_changed.emit(index, tab_text)

    def _init_tab_dashboard_layout(self):
        self.rgb_graphics_view = FitGraphicsView(self)
        self.depth_graphics_view = FitGraphicsView(self)
        self.long_control_widget = LongitudinalControlWidget()
        self.lat_control_widget = LateralControlWidget()
        self.map3d_plot = Map3D()

        # Vision & Position
        navigation_group = QGroupBox("Navigation")
        navigation_layout = QVBoxLayout(navigation_group)
        navigation_layout.addWidget(self.map3d_plot, stretch=2)

        vision_layout = QHBoxLayout()
        vision_layout.addWidget(self.rgb_graphics_view)
        vision_layout.addWidget(self.depth_graphics_view)

        navigation_layout.addLayout(vision_layout, stretch=1)

        # Longitudinal Control
        long_control_group = QGroupBox("Longitudinal Control")
        long_control_layout = QVBoxLayout(long_control_group)
        long_control_layout.addWidget(self.long_control_widget)

        # Lateral Control
        lat_control_group = QGroupBox(title="Lateral Control")
        lat_control_layout = QVBoxLayout(lat_control_group)
        lat_control_layout.addWidget(self.lat_control_widget)

        lat_long_layout = QVBoxLayout()
        lat_long_layout.addWidget(long_control_group)
        lat_long_layout.addWidget(lat_control_group)

        main_layout = QHBoxLayout()
        main_layout.addWidget(navigation_group, stretch=1)
        main_layout.addLayout(lat_long_layout, stretch=1)

        tab1 = QWidget()
        tab1.setLayout(main_layout)
        return tab1

    def _init_tab_sensors_layout(self):
        # Camera
        self.camera_rgb_view = FitGraphicsView(self)
        self.camera_depth_view = FitGraphicsView(self)
        camera_group = QGroupBox(title="Camera")
        camera_layout = QHBoxLayout(camera_group)
        camera_layout.addWidget(self.camera_rgb_view)
        camera_layout.addWidget(self.camera_depth_view)

        # Encoders
        self.left_encoder_plot = EncoderWidget(name="Left")
        self.right_encoder_plot = EncoderWidget(name="Right")
        encoders_group = QGroupBox(title="Encoders")
        encoders_layout = QHBoxLayout(encoders_group)
        encoders_layout.addWidget(self.left_encoder_plot)
        encoders_layout.addWidget(self.right_encoder_plot)

        # IMU
        self.imu3d_plot = IMU3D()
        self.imu_accel_plot = IMURawWidget(title="Acceleration [m/s^2]")
        self.imu_gyro_plot = IMURawWidget(title="Angular Velocity [rad/s]")
        self.imu_mag_plot = IMURawWidget(title="Magnetic Intensity [uT]")
        self.imu_rotation_plot = IMURawWidget(title="Euler Angles [°]")

        imu_rotation_layout = QVBoxLayout()
        imu_rotation_layout.addWidget(self.imu3d_plot, stretch=1)
        imu_rotation_layout.addWidget(self.imu_rotation_plot, stretch=1)

        imu_components_layout = QVBoxLayout()
        imu_components_layout.addWidget(self.imu_accel_plot, stretch=1)
        imu_components_layout.addWidget(self.imu_gyro_plot, stretch=1)
        imu_components_layout.addWidget(self.imu_mag_plot, stretch=1)

        imu_group = QGroupBox(title="IMU")
        imu_layout = QHBoxLayout(imu_group)
        imu_layout.addLayout(imu_rotation_layout)
        imu_layout.addLayout(imu_components_layout)

        layout = QGridLayout()
        layout.addWidget(camera_group, 0, 0)
        layout.addWidget(encoders_group, 1, 0)
        layout.addWidget(imu_group, 0, 1, 2, 2)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)

        tab = QWidget()
        tab.setLayout(layout)

        return tab

    def _init_tab_system_layout(self):
        # self.processor_period_plot = LatencyPlotWidget(name="T Processor", fps_target=30)
        self.processor_dt_plot = LatencyPlotWidget(name="dt Processor", fps_target=30)

        left_layout = QVBoxLayout()
        # left_layout.addWidget(self.processor_period_plot)
        left_layout.addWidget(self.processor_dt_plot)

        layout = QHBoxLayout()
        layout.addLayout(left_layout)

        tab = QWidget()
        tab.setLayout(layout)

        return tab

    def _init_status_bar(self):
        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(False)

        # Latency labels
        latency_header = QLabel("Latency [s] (fps)")
        latency_header.setStyleSheet(f"color: {UIColors.ON_FOREGROUND};")

        self.data_latency_label = EMALatencyLabel(name="Data")
        self.gui_latency_label = EMALatencyLabel(name="GUI")

        latency_labels_widget = QWidget()
        latency_labels_layout = QHBoxLayout(latency_labels_widget)
        latency_labels_layout.setContentsMargins(0, 3, 10, 5)
        latency_labels_layout.addWidget(latency_header)
        latency_labels_layout.addWidget(self.data_latency_label)
        latency_labels_layout.addWidget(self.gui_latency_label)

        status_bar.addPermanentWidget(latency_labels_widget)

    def paintEvent(self, event):
        if hasattr(self, "gui_latency_label"):
            self.gui_latency_label.update()
        super().paintEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F9:
            self.qssdebug.toggle()
            event.accept()
        elif event.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            event.accept()  # Accept the event as it's handled
        elif event.key() == Qt.Key.Key_W:
            self.ui_controller.on_W()
            event.accept()
        elif event.key() == Qt.Key.Key_S:
            self.ui_controller.on_S()
            event.accept()
        elif event.key() == Qt.Key.Key_A:
            self.ui_controller.on_A()
            event.accept()
        elif event.key() == Qt.Key.Key_D:
            self.ui_controller.on_D()
            event.accept()
        else:
            super().keyPressEvent(event)

    def update_simulation_data(self, data: QSimData):
        rgb_pixmap = QPixmap.fromImage(data.rgb_qimage)
        self.rgb_graphics_view.update(rgb_pixmap)

        depth_pixmap = QPixmap.fromImage(data.depth_qimage)
        self.depth_graphics_view.update(depth_pixmap)

        # period_ms = data.processor_period_ns / 1e6
        # self.processor_period_plot.update(dt_ms=period_ms)
        dt_ms = data.raw.original.dt * 1e3
        self.processor_dt_plot.update(dt_ms=dt_ms)

        # TODO: calculate offsets dynamically:

        self.map3d_plot.update_data(
            car_x=0,
            car_y=0,
            car_heading=0,
            car_steering_angle=-data.raw.original.vehicle.steering_angle,
            roadmarks=data.raw.roadmarks_data.roadmarks / 400,
            path=data.raw.roadmarks_data.path / 400,
        )

        # self.long_control_widget.update(
        #     measured_speed=data.raw.original.vehicle.speed,
        #     target_speed=0,
        #     engine_power_percent=0,
        # )
        # self.lat_control_widget.update(
        #     steering_deg=data.raw.original.vehicle.steering_angle, set_steering_deg=0
        # )

    def update_real_data(self, data: QRealData):
        # App Window
        self.data_latency_label.update()

        self.map3d_plot.update_data(
            car_x=0,
            car_y=0,
            car_heading=0,
            car_steering_angle=data.raw.original.control.steering_angle,
            roadmarks=data.raw.roadmarks_data.roadmarks,
            path=data.raw.roadmarks_data.path,
        )

        if hasattr(self, "imu3d_plot"):
            self.imu3d_plot.update_data(
                rotation_quaternion=data.imu_plot.rotation_quaternion,
                accel=data.imu_plot.accel_linear_avg,
                gyro=data.imu_plot.gyro_avg,
            )


class Renderer:
    def __init__(self, profiler=None):
        self.profiler = profiler

    def run(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.app.setStyleSheet(APP_STYLE)
        self.app.setFont(Fonts.GUIMonospace)
        self.app.aboutToQuit.connect(self._stop_threads)

        self.window = RendererMainWindow()
        self.window.init()

        self.t_recorder = RecorderThread()
        self.t_playback = PlaybackThread()
        self.t_sim_data = SimDataThread()
        self.t_real_data = RealDataThread()
        self.threads = [self.t_real_data, self.t_sim_data, self.t_recorder, self.t_playback]
        self.t_sim_data.data_ready.connect(self.window.update_simulation_data)

        self.window.playback_widget.frame_changed.connect(self.t_playback.on_frame_index_set)
        self.window.playback_widget.play_pause_toggled.connect(
            self.t_playback.on_play_pause_toggled
        )
        self.window.playback_widget.start_end_changed.connect(
            self.t_playback.on_start_end_index_change
        )
        self.t_playback.frame_ready.connect(self.window.playback_widget.on_next_frame)
        self.t_playback.record_loaded.connect(self.window.playback_widget.on_record_loaded)

        for t in self.threads:
            t.start()

        self.window.tab_changed.connect(self.on_tab_changed)

        self.t_real_data.long_control_plot_data_ready.connect(
            self.window.long_control_widget.update
        )
        self.t_real_data.lat_control_plot_data_ready.connect(self.window.lat_control_widget.update)
        self.t_sim_data.long_control_plot_data_ready.connect(self.window.long_control_widget.update)
        self.t_sim_data.lat_control_plot_data_ready.connect(self.window.lat_control_widget.update)
        self.t_real_data.camera_rgb_pixmap_ready.connect(self.window.rgb_graphics_view.update)
        self.t_real_data.camera_rgb_updated_pixmap_ready.connect(
            self.window.depth_graphics_view.update
        )
        self.t_real_data.data_ready.connect(self.window.update_real_data)

        self.window.top_tool_bar.record_toggled.connect(self.t_recorder.toggle)
        self.window.records_sidebar.record_selected.connect(self.t_playback.on_record_set)

        # self._autoplay_record()

        if self.profiler is not None:
            self.profiler.stop()
            print(self.profiler.output_text(unicode=True, color=True))

        return self.app.exec()

    @Slot(int, str)
    def on_tab_changed(self, idx, name):
        imu_signals_slots = [
            (self.t_real_data.imu_accel_plot_data_ready, self.window.imu_accel_plot.update),
            (self.t_real_data.imu_gyro_plot_data_ready, self.window.imu_gyro_plot.update),
            (self.t_real_data.imu_mag_plot_data_ready, self.window.imu_mag_plot.update),
            (
                self.t_real_data.imu_rotation_plot_data_ready,
                self.window.imu_rotation_plot.update,
            ),
            (self.t_real_data.lr_encoder_plot_data_ready, self.window.left_encoder_plot.update),
            (self.t_real_data.rr_encoder_plot_data_ready, self.window.right_encoder_plot.update),
            (self.t_real_data.camera_rgb_pixmap_ready, self.window.camera_rgb_view.update),
            (
                self.t_real_data.camera_rgb_updated_pixmap_ready,
                self.window.camera_depth_view.update,
            ),
        ]

        for signal, slot in imu_signals_slots:
            signal.connect(slot) if name == "Sensors" else signal.disconnect(slot)

    def _stop_threads(self):
        pdebug("Shutting down threads gracefully...")

        for t in self.threads:
            t.requestInterruption()

        QApplication.processEvents()

        for t in self.threads:
            try:
                t.stop()
            except Exception as e:
                print(f"Error stopping thread {t}: {e}")

        for t in self.threads:
            if not t.wait(1000):
                print(f"Thread {t} did not quit in time - terminating..")
                t.terminate()
                t.wait()

    def _autoplay_record(self, record: str = "1745511004652250500"):
        self.window.records_sidebar.record_selected.emit(record)
