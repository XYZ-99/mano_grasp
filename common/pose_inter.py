import numpy as np
from .pose_util import random_rotation, unit_quaternion_to_matrix, \
    matrix_to_unit_quaternion, rpy_to_matrix


def cam_pose_to_matrix(cam_pose):
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = cam_pose['rotation']
    pose_mat[:3, 3] = cam_pose['translation']
    return pose_mat

def sapien_cam_pose_to_matrix(cam_pose):
    forward = cam_pose['rotation'].T[0]
    position = cam_pose['look_at'] - forward * cam_pose['distance']
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = cam_pose['rotation']
    pose_mat[:3, 3] = position
    return pose_mat

def get_camera_random_pose(category=None, limited=None, to_matrix=False):
    T =  np.random.randn(3) * 0.05 + [0,0,-0.8]
    rotation = random_rotation()

    cam_pose = {'translation': T, 'rotation': rotation}
    if to_matrix:
        return cam_pose_to_matrix(cam_pose)
    else:
        return cam_pose

def sapien_get_camera_random_pose(category=None, limited=None, to_matrix=False):
    if category == 'laptop':
        look_at_mean = np.array([0.0, 0.0, 0.08])
        look_at_std = np.array([0.05, 0.05, 0.02])
        distance_mean = 1.0
        distance_std = 0.1
    else:
        look_at_mean = np.array([0.0, 0.0, 0.0])
        look_at_std = np.array([0.2, 0.2, 0.2])
        distance_mean = 2
        distance_std = 0.2

    look_at = look_at_mean + np.random.randn(3) * look_at_std
    distance = distance_mean + float(np.random.randn(1)) * distance_std

    def get_uniform(l, r):
        return np.random.rand() * (r - l) + l

    if not limited:
        rotation = random_rotation()
    else:
        if category == 'laptop':
            roll = np.deg2rad(get_uniform(-5, 5))
            pitch = np.deg2rad(get_uniform(5, 30))
            yaw = np.deg2rad(get_uniform(-180, 180))
            rotation = rpy_to_matrix(np.array([roll, pitch, yaw]))
        else:
            assert 0, 'Rotation ranges are not specified for any category'

    cam_pose = {'look_at': look_at, 'distance': distance, 'rotation': rotation}
    if to_matrix:
        return sapien_cam_pose_to_matrix(cam_pose)
    else:
        return cam_pose

def lerp(s, t, alpha):
    return s * (1 - alpha) + t * alpha


def slerp(v0, v1, t_array):
    """Spherical linear interpolation."""
    t_array = np.array([t_array], dtype=float)
    v0 = np.array(v0, dtype=np.float64)
    v1 = np.array(v1, dtype=np.float64)
    dot = np.sum(v0 * v1)

    if dot < 0.0:
        v1 = -v1
        dot = -dot

    DOT_THRESHOLD = 0.999995
    if dot > DOT_THRESHOLD:
        result = v0[np.newaxis, :] + t_array[:, np.newaxis] * (v1 - v0)[np.newaxis, :]
        return (result.T / np.linalg.norm(result, axis=1)).T

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t_array
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    # s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    ret = (s0[:, np.newaxis] * v0[np.newaxis, :]) + (s1[:, np.newaxis] * v1[np.newaxis, :])
    ret /= np.linalg.norm(ret, axis=-1, keepdims=True)
    return ret.reshape(-1)

def sapien_cam_pose_interp(s, t, alpha, to_matrix=False):
    ret = {}
    ret['look_at'] = lerp(s['look_at'], t['look_at'], alpha)
    ret['distance'] = lerp(s['distance'], t['distance'], alpha)

    ret['rotation'] = unit_quaternion_to_matrix(slerp(matrix_to_unit_quaternion(s['rotation']),
                                                      matrix_to_unit_quaternion(t['rotation']), alpha))

    if to_matrix:
        return sapien_cam_pose_to_matrix(ret)
    else:
        return ret

def cam_pose_interp(s, t, alpha, to_matrix=False):
    ret = {}
    ret['translation'] = lerp(s['translation'], t['translation'], alpha)

    ret['rotation'] = unit_quaternion_to_matrix(slerp(matrix_to_unit_quaternion(s['rotation']),
                                                      matrix_to_unit_quaternion(t['rotation']), alpha))

    if to_matrix:
        return cam_pose_to_matrix(ret)
    else:
        return ret


if __name__ == '__main__':
    s = get_camera_random_pose('laptop', False, False)
    t = get_camera_random_pose('laptop', False, False)
