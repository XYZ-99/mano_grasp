import numpy as np
import torch

def rot_diff_rad(rot1, rot2, yaxis_only=False):
    if yaxis_only:
        if isinstance(rot1, np.ndarray):
            y1, y2 = rot1[..., 1], rot2[..., 1]  # [Bs, 3]
            diff = np.sum(y1 * y2, axis=-1)  # [Bs]
            diff = np.clip(diff, a_min=-1.0, a_max=1.0)
            return np.arccos(diff)
        else:
            y1, y2 = rot1[..., 1], rot2[..., 1]  # [Bs, 3]
            diff = torch.sum(y1 * y2, dim=-1)  # [Bs]
            diff = torch.clamp(diff, min=-1.0, max=1.0)
            return torch.acos(diff)
    else:
        if isinstance(rot1, np.ndarray):
            mat_diff = np.matmul(rot1, rot2.swapaxes(-1, -2))
            diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
            diff = (diff - 1) / 2.0
            diff = np.clip(diff, a_min=-1.0, a_max=1.0)
            return np.arccos(diff)
        else:
            mat_diff = torch.matmul(rot1, rot2.transpose(-1, -2))
            diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
            diff = (diff - 1) / 2.0
            diff = torch.clamp(diff, min=-1.0, max=1.0)
            return torch.acos(diff)


def rot_diff_degree(rot1, rot2, yaxis_only=False):
    return rot_diff_rad(rot1, rot2, yaxis_only=yaxis_only) / np.pi * 180.0


def normalize(q):
    assert q.shape[-1] == 4
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / norm


def assert_normalized(q, atol=1e-3):
    assert q.shape[-1] == 4
    norm = np.linalg.norm(q, axis=-1)
    norm_check = np.abs(norm - 1.0)
    try:
        assert np.max(norm_check) < atol
    except:
        print("normalization failure: {}.".format(np.max(norm_check)))
        return -1
    return 0


def unit_quaternion_to_matrix(q):
    assert_normalized(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    matrix = np.stack((1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
                       2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w,
                       2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y),
                      axis=-1)
    matrix_shape = list(matrix.shape)[:-1] + [3, 3]
    return matrix.reshape(matrix_shape)


def matrix_to_unit_quaternion(matrix):
    assert matrix.shape[-1] == matrix.shape[-2] == 3

    trace = 1.0 + matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    trace = np.clip(trace, a_min=0., a_max=None)

    r = np.sqrt(trace)
    if r == 0.0:
        s = 1e6
    else:
        s = 1.0 / (2 * r)
    w = 0.5 * r
    x = (matrix[..., 2, 1] - matrix[..., 1, 2]) * s
    y = (matrix[..., 0, 2] - matrix[..., 2, 0]) * s
    z = (matrix[..., 1, 0] - matrix[..., 0, 1]) * s

    q = np.stack((w, x, y, z), axis=-1)

    return normalize(q)


def axis_theta_to_quater(axis, theta):  # axis: [Bs, 3], theta: [Bs]
    w = np.cos(theta / 2.)  # [Bs]
    u = np.sin(theta / 2.)  # [Bs]
    xyz = axis * np.expand_dims(u, -1)  # [Bs, 3]
    new_q = np.concatenate([np.expand_dims(w, -1), xyz], axis=-1)  # [Bs, 4]
    new_q = normalize(new_q)
    return new_q


def quater_to_axis_theta(quater):
    quater = normalize(quater)
    cosa = quater[..., 0]
    sina = np.sqrt(1 - cosa ** 2)
    norm = np.expand_dims(sina, -1)
    mask = (norm < 1e-8).astype(float)
    axis = quater[..., 1:] / np.maximum(norm, mask)
    theta = 2 * np.arccos(np.clip(cosa, a_min=-1, a_max=1))
    return axis, theta


def axis_theta_to_matrix(axis, theta):
    quater = axis_theta_to_quater(axis, theta)  # [Bs, 4]
    return unit_quaternion_to_matrix(quater)


def matrix_to_axis_theta(matrix):
    quater = matrix_to_unit_quaternion(matrix)
    return quater_to_axis_theta(quater)


def matrix_to_rotvec(matrix):
    axis, theta = matrix_to_axis_theta(matrix)
    theta = theta % (2 * np.pi) + 2 * np.pi
    return axis * np.expand_dims(theta, -1)


def rotvec_to_matrix(rotvec):  # [Bs, 3]
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)  # [Bs, 1]
    mask = (theta < 1e-8).astype(float)
    axis = rotvec / np.maximum(theta, mask)  # [Bs, 3]
    theta = theta.squeeze(-1)  # [Bs]
    return axis_theta_to_matrix(axis, theta)


def random_rotation():
    quat = np.random.randn(4)
    quat = quat / np.linalg.norm(quat)
    return unit_quaternion_to_matrix(quat)


def rpy_to_matrix(rpy):  # [Bs, 3]
    """
    roll: around x (forward)
    pitch: around y (left)
    yaw: around z (up)
    extrinsic
    """
    x, y, z = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cosx, sinx = np.cos(x), np.sin(x)
    cosy, siny = np.cos(y), np.sin(y)
    cosz, sinz = np.cos(z), np.sin(z)
    mat = np.stack([cosz * cosy, cosz * siny * sinx - sinz * cosx, cosz * siny * cosx + sinz * sinx,
                    sinz * cosy, sinz * siny * sinx + cosz * cosx, sinz * siny * cosx - cosz * siny,
                    -siny, cosy * sinx, cosy * cosx], axis=-1)
    shape = mat.shape[:-1] + (3, 3)
    mat = compute_rotation_matrix_from_matrix(mat.reshape(3, 3))
    mat = mat.reshape(shape)
    return mat


def compute_rotation_matrix_from_matrix(matrices):  # let's just deal with 3 x 3 matrices..
    def proj_u_a(u, a):  # [3], [3]
        factor = np.dot(u, a) / np.maximum(np.dot(u, u), 1e-8)
        return factor * u

    def normalize_vector(u):
        return u / np.maximum(1e-8, np.sqrt(np.dot(u, u)))

    a1 = matrices[:, 0]
    a2 = matrices[:, 1]
    a3 = matrices[:, 2]

    u1 = a1
    u2 = a2 - proj_u_a(u1, a2)
    u3 = a3 - proj_u_a(u1, a3) - proj_u_a(u2, a3)

    e1 = normalize_vector(u1)
    e2 = normalize_vector(u2)
    e3 = normalize_vector(u3)

    rmat = np.stack([e1, e2, e3], axis=1)

    return rmat



def main():
    import sys
    import os

    base_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(base_dir, '..'))
    sys.path.append(os.path.join(base_dir, '..', '..'))
    sys.path.append(os.path.join(base_dir, '..', '..', '..'))

    from misc.visualize.vis_utils import plot_arrows
    point = np.zeros((1, 3))
    mat = rpy_to_matrix(np.deg2rad(np.array([10, 20, 0])))
    plot_arrows(point, mat)


if __name__ == '__main__':
    main()
    