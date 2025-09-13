import numpy as np
import cv2

from typing import Any, Iterable, Tuple
from time import time_ns

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QQuaternion

from modules.messaging import messaging
from datalink.data import ProcessedSimData, ProcessedRealData, JPGImageData, IMU2Data
from cv2 import imdecode, IMREAD_COLOR

from modules.ui.plots import DATA_QUEUE_SIZE, ENCODER_RAW2DEG

# Helpers
# -------------------------------------------------------------------------------------------------


def jpg2qimg(jpg: bytes):
    bgr = imdecode(np.frombuffer(jpg, np.uint8), IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb2qimg(rgb)


def rgb2qimg(rgb: np.ndarray[Any, np.dtype[np.uint8]]):
    h, w, channels = rgb.shape
    return QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888)


def depth2qimg(depth: np.ndarray) -> QImage:
    depth_colormap = depth2colormap(depth)
    w, h, _ = depth_colormap.shape
    return QImage(depth_colormap.data, h, w, QImage.Format_BGR888)


def depth2colormap(depth: np.ndarray):
    normalized_inverted = 255 - (depth / 5000 * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(normalized_inverted, cv2.COLORMAP_INFERNO)
    return colormap


# Data
# -------------------------------------------------------------------------------------------------


class LongControlPlotData:
    def __init__(self):
        self.measured_speed = np.zeros(DATA_QUEUE_SIZE)
        self.target_speed = np.zeros(DATA_QUEUE_SIZE)
        self.engine_power_percent = np.zeros(DATA_QUEUE_SIZE)

    def update(self, avg_speed, target_speed, engine_power_percent):
        self.measured_speed[:-1] = self.measured_speed[1:]
        self.measured_speed[-1] = avg_speed

        self.target_speed[:-1] = self.target_speed[1:]
        self.target_speed[-1] = target_speed

        self.engine_power_percent[:-1] = self.engine_power_percent[1:]
        self.engine_power_percent[-1] = engine_power_percent


class LatControlPlotData:
    def __init__(self):
        self.estimated_sa = np.zeros(DATA_QUEUE_SIZE)
        self.target_sa = np.zeros(DATA_QUEUE_SIZE)
        self.input_sa = np.zeros(DATA_QUEUE_SIZE)

    def update(self, estimated_sa, target_sa, input_sa):
        self.estimated_sa[:-1] = self.estimated_sa[1:]
        self.estimated_sa[-1] = estimated_sa

        self.target_sa[:-1] = self.target_sa[1:]
        self.target_sa[-1] = target_sa

        self.input_sa[:-1] = self.input_sa[1:]
        self.input_sa[-1] = input_sa


class EncoderPlotData:
    HISTORY_SIZE = 15

    def __init__(self):
        self.radial_xs = np.zeros(self.HISTORY_SIZE)
        self.radial_ys = np.zeros(self.HISTORY_SIZE)

        self.angle_deg = 0
        self.angle_deg_change = 0

        self.avg_magnitude = 0
        self.avg_magnitude_change = 0

        self._last_angle_deg = 0
        self._last_magnitude = 0

    def update(self, readings: Iterable):
        angle_deg_changes = []
        magnitude_changes = []
        magnitude_sum = 0

        for data in readings:
            angle_deg = data.position * ENCODER_RAW2DEG
            angle_deg_change = (angle_deg - self._last_angle_deg + 180) % 360 - 180
            angle_deg_changes.append(angle_deg_change)
            self._last_angle_deg = angle_deg

            magnitude_change = data.magnitude - self._last_magnitude
            magnitude_changes.append(magnitude_change)
            magnitude_sum += data.magnitude
            self._last_magnitude = data.magnitude

        self.angle_deg = self._last_angle_deg
        angle = np.deg2rad(self._last_angle_deg)

        self.radial_xs[:-1] = self.radial_xs[1:]
        self.radial_xs[-1] = self._last_magnitude * np.cos(angle)
        self.radial_ys[:-1] = self.radial_ys[1:]
        self.radial_ys[-1] = self._last_magnitude * np.sin(angle)

        self.angle_deg_change = sum(angle_deg_changes)
        self.avg_magnitude = magnitude_sum / len(readings) if len(readings) else 0
        self.avg_magnitude_change = (
            sum(magnitude_changes) / len(magnitude_changes) if len(readings) else 0
        )


class IMURawPlotData:
    def __init__(self):
        self.data = np.zeros((3, DATA_QUEUE_SIZE))

    def update(self, xyz: Tuple[float, float, float]):
        self.data[:, :-1] = self.data[:, 1:]
        self.data[:, -1] = xyz


class IMUPlotData:
    def __init__(self):
        self.rotation_quaternion = QQuaternion()
        self.rotation_euler_deg = (0, 0, 0)
        self.rotation_euler = (0, 0, 0)
        self.accel_linear_avg = (0, 0, 0)
        self.gyro_avg = (0, 0, 0)
        self.mag_avg = (0, 0, 0)

        self.rotation_euler_history = IMURawPlotData()
        self.rotation_euler_deg_history = IMURawPlotData()
        self.accel_linear_avg_history = IMURawPlotData()
        self.gyro_avg_history = IMURawPlotData()
        self.mag_avg_history = IMURawPlotData()

    def update(self, readings: Iterable[IMU2Data]):
        if not len(readings):
            return

        self.accel_linear_avg = np.average([r.accel_linear for r in readings], axis=0)
        self.gyro_avg = np.average([r.gyro for r in readings], axis=0)
        self.mag_avg = np.average([r.mag for r in readings], axis=0)

        self.rotation_quaternion = QQuaternion(*readings[-1].rotation_quaternion)
        self.rotation_euler_deg = readings[-1].rotation_euler_deg
        self.rotation_euler = np.deg2rad(self.rotation_euler_deg)

        self.accel_linear_avg_history.update(self.accel_linear_avg)
        self.gyro_avg_history.update(self.gyro_avg)
        self.mag_avg_history.update(self.mag_avg)
        self.rotation_euler_history.update(self.rotation_euler)
        self.rotation_euler_deg_history.update(self.rotation_euler_deg)


class QSimData:
    def __init__(self, raw: ProcessedSimData):
        self.raw = raw
        self.rgb_qimage: QImage = None
        self.depth_qimage: QImage = None
        self.processor_period_ns: float = 0
        self.processor_dt_ns: float = 0


class QRealData:
    def __init__(self, raw: ProcessedRealData):
        self.raw = raw
        self.processor_period_ns: float = 0
        self.processor_dt_ns: float = 0

        self.camera_rgb_pixmap: QPixmap
        self.camera_rgb_updated_pixmap: QPixmap

        self.long_control_plot: LongControlPlotData
        self.lat_control_plot: LatControlPlotData

        self.lr_encoder_plot: EncoderPlotData
        self.rr_encoder_plot: EncoderPlotData

        self.imu_plot: IMUPlotData


# Processors
# -------------------------------------------------------------------------------------------------


class SimDataThread(QThread):
    data_ready = Signal(QSimData)
    long_control_plot_data_ready = Signal(LongControlPlotData)
    lat_control_plot_data_ready = Signal(LatControlPlotData)

    def __init__(self):
        super().__init__()

        self.long_control_plot_data = LongControlPlotData()
        self.lat_control_plot_data = LatControlPlotData()

    def run(self):
        q = messaging.q_sim_processing.get_consumer()
        last_put_timestamp = time_ns()
        self._is_running = True

        while self._is_running and not self.isInterruptionRequested():
            data: ProcessedSimData = q.get(100)
            if data is None:
                continue
            qsim_data = QSimData(raw=data)
            qsim_data.rgb_qimage = rgb2qimg(data.debug_image)
            qsim_data.depth_qimage = depth2qimg(data.depth)
            qsim_data.processor_period_ns = q.last_put_timestamp - last_put_timestamp
            qsim_data.processor_dt_ns = q.last_put_timestamp - data.begin_timestamp
            last_put_timestamp = q.last_put_timestamp

            self.long_control_plot_data.update(
                data.original.vehicle.speed / 100,
                data.control_data.speed * 20 * 20,
                data.original.vehicle.speed / 22,
            )
            self.long_control_plot_data_ready.emit(self.long_control_plot_data)

            self.lat_control_plot_data.update(
                data.original.vehicle.steering_angle,
                data.control_data.steering_angle,
                data.original.vehicle.steering_angle,
            )
            self.lat_control_plot_data_ready.emit(self.lat_control_plot_data)

            self.data_ready.emit(qsim_data)

    def stop(self):
        self._is_running = False
        self.quit()


def estimate_steering_angle_deg(speed, yaw_rate, wheelbase=0.185):
    """
    Bicycle model Steering Angle estimate

    Args:
        speed: Vehicle speed [m/s]
        yaw_rate: Vehicle yaw rate [rad/s]
        wheelbase: Distance between front and rear axles [m]

    Returns:
        Estimated Steering Angle in degrees
    """
    if abs(speed) < 0.1:
        return 0.0

    # Bicycle model: tan(δ) = (L * ω) / v
    # where: δ = steering angle, L = wheelbase, ω = yaw rate, v = speed
    steering_angle_rad = np.arctan2(wheelbase * yaw_rate, speed)

    # Convert to degrees
    return np.degrees(steering_angle_rad)


class RealDataThread(QThread):
    data_ready = Signal(QRealData)

    long_control_plot_data_ready = Signal(LongControlPlotData)
    lat_control_plot_data_ready = Signal(LatControlPlotData)
    camera_rgb_pixmap_ready = Signal(QPixmap)
    camera_rgb_updated_pixmap_ready = Signal(QPixmap)
    lr_encoder_plot_data_ready = Signal(EncoderPlotData)
    rr_encoder_plot_data_ready = Signal(EncoderPlotData)
    imu_accel_plot_data_ready = Signal(IMURawPlotData)
    imu_gyro_plot_data_ready = Signal(IMURawPlotData)
    imu_mag_plot_data_ready = Signal(IMURawPlotData)
    imu_rotation_plot_data_ready = Signal(IMURawPlotData)

    def __init__(self):
        super().__init__()

        self.long_control_plot_data = LongControlPlotData()
        self.lat_control_plot_data = LatControlPlotData()

        self.lr_encoder_plot_data = EncoderPlotData()
        self.rr_encoder_plot_data = EncoderPlotData()

        self.imu_plot_data = IMUPlotData()

    def run(self):
        q = messaging.q_real_processing.get_consumer()
        self._is_running = True

        while self._is_running and not self.isInterruptionRequested():
            pr_data: ProcessedRealData = q.get(100)
            if pr_data is None:
                continue

            jpg_image_data: JPGImageData = pr_data.original.sensor_fusion.camera
            image_array = imdecode(np.frombuffer(jpg_image_data.jpg, np.uint8), IMREAD_COLOR)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            tgt_speed = 1.25 if pr_data.original.control.speed > 0 else pr_data.original.control.speed
            self.long_control_plot_data.update(
                pr_data.original.sensor_fusion.avg_speed,
                tgt_speed,
                pr_data.original.actuators.motor_power * 100,
            )
            self.long_control_plot_data_ready.emit(self.long_control_plot_data)

            estimated_sa = estimate_steering_angle_deg(
                speed=pr_data.original.sensor_fusion.avg_speed,
                yaw_rate=self.imu_plot_data.gyro_avg[2],
            )

            self.lat_control_plot_data.update(
                estimated_sa=estimated_sa,
                target_sa=pr_data.control_data.steering_angle / 3,
                input_sa=pr_data.control_data.steering_angle / 3,
            )
            self.lat_control_plot_data_ready.emit(self.lat_control_plot_data)

            encoder_readings = [x.encoder_data for x in pr_data.original.sensor_fusion.speedometer]
            self.lr_encoder_plot_data.update(encoder_readings)
            self.lr_encoder_plot_data_ready.emit(self.lr_encoder_plot_data)
            self.rr_encoder_plot_data.update(encoder_readings)
            self.rr_encoder_plot_data_ready.emit(self.rr_encoder_plot_data)

            imu_readings = pr_data.original.sensor_fusion.imu
            self.imu_plot_data.update(imu_readings)
            self.imu_accel_plot_data_ready.emit(self.imu_plot_data.accel_linear_avg_history.data)
            self.imu_gyro_plot_data_ready.emit(self.imu_plot_data.gyro_avg_history.data)
            self.imu_mag_plot_data_ready.emit(self.imu_plot_data.mag_avg_history.data)
            self.imu_rotation_plot_data_ready.emit(
                self.imu_plot_data.rotation_euler_deg_history.data
            )

            camera_rgb_pixmap = QPixmap.fromImage(rgb2qimg(image_array))
            self.camera_rgb_pixmap_ready.emit(camera_rgb_pixmap)
            camera_debug_rgb_pixmap = QPixmap.fromImage(rgb2qimg(pr_data.debug_image))
            self.camera_rgb_updated_pixmap_ready.emit(camera_debug_rgb_pixmap)

            qreal_data = QRealData(pr_data)
            # qreal_data.lat_control_plot = self.lat_control_plot_data
            # qreal_data.long_control_plot = self.long_control_plot_data
            # qreal_data.lr_encoder_plot = self.lr_encoder_plot_data
            # qreal_data.rr_encoder_plot = self.rr_encoder_plot_data
            qreal_data.imu_plot = self.imu_plot_data
            self.data_ready.emit(qreal_data)

    def stop(self):
        self._is_running = False
        self.quit()
