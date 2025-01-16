import numpy as np

def translation_matrix(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def rotation_matrix(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    R_homogeneous = np.eye(4)
    R_homogeneous[:3, :3] = R
    return R_homogeneous

def transformation_matrix(*args):
    r = np.eye(4)
    for m in args:
        r = r @ m
    return r

if __name__ == "__main__":
    T = translation_matrix(10, 20, 30)
    R = rotation_matrix(np.deg2rad(0), np.deg2rad(-15), np.deg2rad(0))
    H = transformation_matrix(R, T)
    H_inv = np.linalg.inv(H)

    point = np.array([0, 0, 0, 1])
    trans_point = H @ point
    orig_point = H_inv @ trans_point

    print(point)
    print(trans_point)
    print(orig_point)