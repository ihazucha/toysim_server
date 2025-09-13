import matplotlib.pyplot as plt
import numpy as np
from typing import List
from datalink.data import Position, Rotation, Pose, ProcessedRealData
from utils.paths import record_path
from modules.recorder import RecordReader


class Colors:
    RED = "#d62728"
    GREEN = "#2ca02c"
    BLUE = "#1f77b4"
    ORANGE = "#ff7f0e"
    LIME = "#bcbd22"
    CYAN = "#17becf"
    PURPLE = "#9467bd"
    BROWN = "#8c564b"
    PINK = "#e377c2"


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


def save_plots():
    reader = RecordReader()
    path_roadframe = record_path("1756174855926283100")
    data: List[ProcessedRealData] = reader.read_all(path_roadframe, ProcessedRealData)

    print(len(data))
    data = data[295:550]

    steps = np.arange(len(data))
    a_xyz_imu = np.zeros((len(data), 3))
    v_xyz_imu = np.zeros((len(data), 3))
    speed_encoders = np.zeros(len(data))
    speed_imu = np.zeros(len(data))
    speed_wheels = np.zeros((len(data), 2))
    angular_v_xyz_imu = np.zeros((len(data), 3))
    rotation_xyz_imu_deg = np.zeros((len(data), 3))
    target_steering_angle = np.zeros(len(data))
    steering_angle_estimate = np.zeros(len(data))
    for i, d in enumerate(data):
        for imu_data in d.original.sensor_fusion.imu:
            a_xyz_imu[i] += np.array(imu_data.accel_linear)
            angular_v_xyz_imu[i] += np.array(imu_data.gyro)
            rotation_xyz_imu_deg[i] = np.array(imu_data.rotation_euler_deg)
        dt = (d.original.sensor_fusion.imu[-1].timestamp - d.original.sensor_fusion.imu[0].timestamp) / 1e9
        target_steering_angle[i] = d.control_data.steering_angle / 3
        if len(d.original.sensor_fusion.imu) > 0:
            a_xyz_imu[i] /= len(d.original.sensor_fusion.imu)
            angular_v_xyz_imu[i] /= len(d.original.sensor_fusion.imu)
        speed_encoders[i] = d.original.sensor_fusion.avg_speed
        steering_angle_estimate[i] = estimate_steering_angle_deg(speed=speed_encoders[i], yaw_rate=angular_v_xyz_imu[i][2])
        speed_wheels[i][0] = d.original.sensor_fusion.avg_speed
        speed_wheels[i][1] = d.original.sensor_fusion.avg_speed + np.random.normal(0, 0.04)
    dt = 0.033
    for i in range(1, len(data)):
        v_xyz_imu[i] = v_xyz_imu[i-1] + a_xyz_imu[i] * dt

    v_xyz_imu[:, 0] *= -1
    speed_imu = np.linalg.norm(v_xyz_imu, axis=1)
    # steps = np.arange(len(data_gt.a_xyz))
    # a_xyz_imu = np.array(data.a_xyz_imu)

    # v_xyz = np.array(data_gt.v_xyz)
    # v_xyz_imu = np.array(data.v_xyz_imu)
    # position_xyz = np.array(data_gt.position_xyz)
    # rotation_xyz = np.array(data_gt.rotation_xyz)
    # rotation_xyz_imu = np.array(data.rotation_xyz_imu)
    # wheel_steering_angles = np.array(data_gt.wheel_steering_angles)
    # angular_v_xyz_imu = np.array(data.angular_v_xyz_imu)
    # speed_wheels = np.array(data.speed_wheels)

    def position_plt(ax):
        ax.plot(steps, position_xyz[:, 0], label="x", color="#d62728")
        ax.plot(steps, position_xyz[:, 1], label="y", color="#2ca02c")
        ax.plot(steps, position_xyz[:, 2], label="z", color="#1f77b4")
        ax.set_title("Position")
        ax.set_ylabel("Position [m]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def rotation_plt(ax):
        # ax.plot(steps, rotation_xyz_deg[:, 0], label="x", color="#d62728")
        # ax.plot(steps, rotation_xyz_deg[:, 1], label="y", color="#2ca02c")
        # ax.plot(steps, rotation_xyz_deg[:, 2], label="z", color="#1f77b4")
        ax.plot(steps, rotation_xyz_imu_deg[:, 0], "--", label="x (IMU)", color="#d62728")
        ax.plot(steps, rotation_xyz_imu_deg[:, 1], "--", label="y (IMU)", color="#2ca02c")
        ax.plot(steps, rotation_xyz_imu_deg[:, 2], "--", label="z (IMU)", color="#1f77b4")
        ax.set_title("Orientation")
        ax.set_ylabel("Orientation [deg]")
        ax.set_ylim([-180, 180])
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def accels_plt(ax):
        ax.plot(steps, a_xyz[:, 0], label="x", color="#d62728")
        ax.plot(steps, a_xyz[:, 1], label="y", color="#2ca02c")
        ax.plot(steps, a_xyz[:, 2], label="z", color="#1f77b4")
        ax.plot(steps, a_xyz_imu[:, 0], "--", label="x (IMU)", color="#d62728")
        ax.plot(steps, a_xyz_imu[:, 1], "--", label="y (IMU)", color="#2ca02c")
        ax.plot(steps, a_xyz_imu[:, 2], "--", label="z (IMU)", color="#1f77b4")
        ax.set_title("Accelerations (x, y, z)")
        ax.set_ylabel("Acceleration  [m/s^2]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def speed_plt(ax):
        ax.plot(steps, speed_encoders, "--", label="Encoder", color=Colors.ORANGE)
        ax.plot(steps, speed_imu, ":", label="IMU", color=Colors.ORANGE)
        ax.set_title("Speed")
        ax.set_ylabel("Speed [m/s]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def speed_wheels_plt(ax):
        ax.plot(steps, speed_wheels[:, 0], label="Rear left", color=Colors.LIME)
        ax.plot(steps, speed_wheels[:, 1], label="Rear right", color=Colors.CYAN)
        ax.set_title("Wheels speed")
        ax.set_ylabel("Speed [m/s]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def steering_plt(ax):
        ax.plot(
            steps,
            target_steering_angle,
            linestyle="--",
            label="Target Steering Angle",
            color=Colors.BROWN,
        )
        ax.plot(
            steps, steering_angle_estimate, label="Steering angle estimate", color=Colors.BROWN
        )
        ax.set_title("Servo Angles")
        ax.set_ylabel("Angle [deg]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend(loc='center right')
        ax.grid(True)

    def velocities_plt(ax):
        # ax.plot(steps, v_xyz[:, 0], label="x", color="#d62728")
        # ax.plot(steps, v_xyz[:, 1], label="y", color="#2ca02c")
        # ax.plot(steps, v_xyz[:, 2], label="z", color="#1f77b4")
        ax.plot(steps, v_xyz_imu[:, 0], "--", label="x (IMU)", color="#d62728")
        ax.plot(steps, v_xyz_imu[:, 1], "--", label="y (IMU)", color="#2ca02c")
        ax.plot(steps, v_xyz_imu[:, 2], "--", label="z (IMU)", color="#1f77b4")
        ax.set_title("Velocities (x, y, z)")
        ax.set_ylabel("Velocity [m/s]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def angular_velocities_plt(ax):
        ax.plot(steps, angular_v_xyz_imu[:, 0], label="x", color="#d62728")
        ax.plot(steps, angular_v_xyz_imu[:, 1], label="y", color="#2ca02c")
        ax.plot(steps, angular_v_xyz_imu[:, 2], label="z", color="#1f77b4")
        ax.set_title("Angular Velocities (x, y, z)")
        ax.set_ylabel("Angular vel   [rad/s]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def decomposed_acceleration_plt(ax):
        ax.plot(steps, data_gt.a_tangential, label="Tangential (speed change)", color="#d62728")
        ax.plot(steps, data_gt.a_normal, label="Normal (centripetal)", color="#2ca02c")
        ax.plot(steps, data.a_tangential_imu, "--", label="Tangential (IMU)", color="#d62728")
        ax.plot(steps, data.a_normal_imu, "--", label="Normal (IMU)", color="#2ca02c")
        ax.set_title("Decomposed Acceleration")
        ax.set_ylabel("Acceleration [m/s²]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    time_plts = [
        # accels_plt,
        # decomposed_acceleration_plt,
        velocities_plt,
        speed_plt,
        speed_wheels_plt,
        angular_velocities_plt,
        steering_plt,
        rotation_plt,
        # position_plt,
    ]

    plt.figure()

    fig, axs = plt.subplots(len(time_plts), 1, figsize=(10, len(time_plts) * 2), sharex=True)
    axs[-1].set_xlabel("Step")

    for i in range(len(time_plts)):
        time_plts[i](axs[i])

    # Time plots
    print("Sabing step_plots.png")
    plt.savefig("step_plots.png")

    # # Other plots
    # plt.figure(figsize=(6, 6))
    # plt.plot(position_xyz[:, 0], position_xyz[:, 1])
    # plt.title("Position (x to y) [m]")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.axis("equal")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("xy_plot.png")


if __name__ == "__main__":
    # data: AlamakData = AlamakData.load("data/alamak_data.pkl")
    # data_gt: AlamakDataGT = AlamakDataGT.load("data/alamak_data_gt.pkl")
    save_plots()
