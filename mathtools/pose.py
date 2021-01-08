import logging

import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt


logger = logging.getLogger()


class RigidTransform(object):
    def __init__(self, translation=None, rotation=None):
        self._t = translation
        self._r = rotation

    def __eq__(self, other):
        translations_equal = np.allclose(self._t, other._t)
        rotations_equal = np.allclose(self._r.as_quat(), other._r.as_quat())
        return translations_equal and rotations_equal

    def apply(self, vectors, inverse=False):
        if inverse:
            return self.inv().apply(vectors)
        return self._r.apply(vectors) + self._t

    def inv(self):
        inv_r = self._r.inv()
        inv_t = inv_r.apply(-self._t)
        return RigidTransform(inv_t, inv_r)

    def __mul__(self, other):
        new_r = self._r * other._r
        new_t = self._r.apply(other._t) + self._t
        return RigidTransform(new_t, new_r)

    def __truediv__(self, other):
        return other.inv() * self

    def __repr__(self):
        rpy_angles = self._r.as_euler('zyx', degrees=True)
        return f"RigidTransform(rpy: {rpy_angles},  xyz: {self._t})"

    def getAttributes(self, rotation_as=None):
        if rotation_as is None:
            r = self._r
        if rotation_as == 'quaternion':
            r = self._r.as_quat()
        elif rotation_as == 'euler':
            r = self._r.as_euler()
        else:
            raise NotImplementedError()

        return self._t, r

    def mean(self, weights=None):
        if not self._t.any():
            t_mean = np.full(3, np.nan)
        else:
            t_mean = np.average(self._t, weights=weights, axis=0)

        r_mean = self._r.mean(weights=weights)

        return RigidTransform(t_mean, r_mean)

    def magnitude(self):
        t_mag = np.linalg.norm(self._t, axis=1)
        r_mag = self._r.magnitude()
        return t_mag, r_mag

    @staticmethod
    def identity():
        """ Return the identity transform. """
        r = Rotation.identity()
        t = np.zeros(3)
        return RigidTransform(t, r)


def unpack_pose(pose_seq, ignore_nan=False):
    if ignore_nan:
        row_has_nan = np.isnan(pose_seq).any(axis=1)
        return unpack_pose(pose_seq[~row_has_nan, :], ignore_nan=False)

    t = pose_seq[:, 0:3]
    r = Rotation.from_quat(pose_seq[:, 3:7])
    return RigidTransform(t, r)


def pack_pose(transform):
    packed = np.hstack(transform.getAttributes(rotation_as='quaternion'))
    return packed


def relPose(lhs, rhs, magnitude_only=False, rotation_as='quaternion'):
    """ lhs - rhs := inv(rhs) o (lhs) """

    if np.isnan(lhs).any() or np.isnan(rhs).any():
        either_row_has_nan = np.isnan(lhs).any(axis=1) + np.isnan(rhs).any(axis=1)
        non_nan_rel_pose = relPose(
            lhs[~either_row_has_nan, :], rhs[~either_row_has_nan, :],
            magnitude_only=magnitude_only, rotation_as=rotation_as
        )
        rel_pose = np.full(lhs.shape[0:1] + non_nan_rel_pose.shape[1:], np.nan)
        rel_pose[~either_row_has_nan, ...] = non_nan_rel_pose
        return rel_pose

    lhs = unpack_pose(lhs)
    rhs = unpack_pose(rhs)
    quotient = (lhs / rhs)

    if magnitude_only:
        rel_pose = np.column_stack(quotient.magnitude())
    else:
        rel_pose = np.hstack(quotient.getAttributes(rotation_as=rotation_as))

    return rel_pose


def avgPose(*pose_seqs):
    avg_poses = np.full(pose_seqs[0].shape, np.nan)

    all_poses = np.stack(pose_seqs, axis=1)
    for i, sample_poses in enumerate(all_poses):
        transforms = unpack_pose(sample_poses, ignore_nan=True)
        avg_transform = transforms.mean()
        avg_pose = pack_pose(avg_transform)
        avg_poses[i, :] = avg_pose

    return avg_poses


def plotDiffs(lhs, rhs, fn):
    diff = relPose(lhs, rhs, magnitude_only=False, angle_type='euler')

    num_rows = diff.shape[1]
    num_cols = 2
    figsize = (8 * num_cols, 3 * num_rows)
    f, axes = plt.subplots(num_rows, num_cols, sharex=True, figsize=figsize)

    for i in range(num_rows):
        axes[i, 0].plot(lhs[:, i])
        axes[i, 0].plot(rhs[:, i])
        axes[i, 1].plot(diff[:, i])

    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def plotPoses(fn, *poses, labels=None):
    num_rows = poses[0].shape[1]
    figsize = (12, 3 * num_rows)
    f, axes = plt.subplots(num_rows, sharex=True, figsize=figsize)
    if num_rows == 1:
        axes = [axes]

    for i in range(num_rows):
        for j, pose_seq in enumerate(poses):
            if labels is None:
                label = None
            else:
                label = labels[j]
            axes[i].plot(pose_seq[:, i], label=label)

        if labels is not None:
            axes[i].legend()

    plt.tight_layout()
    plt.savefig(fn)
    plt.close()
