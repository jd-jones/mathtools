import sys
import functools

import numpy as np
import torch
from torch import (
    arange, meshgrid,
    diag, einsum, sum, unique,
    cos, sin, cross, dot,
    floor, ceil,
)

import kornia


THIS_MODULE = sys.modules[__name__]


DEFAULT_DEVICE = None


# TODO: logical_not, append


def ensure_tensor(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            args = tuple(array(a) for a in args)
            return func(*args, **kwargs)
    return wrapped


def any(x):
    return x.byte().any()


def all(x):
    return x.byte().all()


def array(obj):
    return torch.tensor(obj, device=DEFAULT_DEVICE)


def zeros(*args, **kwargs):
    return torch.zeros(*args, **kwargs, device=DEFAULT_DEVICE)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs, device=DEFAULT_DEVICE)


def ones(*args, **kwargs):
    return torch.ones(*args, **kwargs, device=DEFAULT_DEVICE)


def ones_like(*args, **kwargs):
    return torch.ones_like(*args, **kwargs, device=DEFAULT_DEVICE)


def eye(*args, **kwargs):
    return torch.eye(*args, **kwargs, device=DEFAULT_DEVICE)


def loadtxt(fn, **kwargs):
    a = np.loadtxt(fn, **kwargs)
    return array(a)


def diff(a, n=1, axis=-1, prepend=None, append=None):
    if n != 1:
        raise NotImplementedError()

    if prepend is not None:
        raise NotImplementedError()

    if append is not None:
        raise NotImplementedError()

    if axis == 0:
        return a[1:] - a[:-1]
    elif axis == -1:
        return a[..., 1:] - a[..., :-1]
    else:
        raise NotImplementedError()


def ravel(a):
    # FIXME: this never copies to contiguous memory, which might be necessary.
    return a.view(-1)


def radians(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float)

    return kornia.conversions.deg2rad(x)


def hstack(arrays):
    return torch.cat(arrays, dim=1)


def vstack(arrays):
    return torch.cat(arrays, dim=0)


def dstack(arrays):
    raise NotImplementedError()


def column_stack(arrays):
    raise NotImplementedError()


def row_stack(arrays):
    raise NotImplementedError()


def repeat():
    raise NotImplementedError()


def expand_dims(array, axis=None):
    return array.unsqueeze(axis)


def pinv(array):
    return torch.pinverse(array)


def transpose(array):
    return array.transpose(0, 1)


def arctan2(x):
    return torch.atan2(x)


# Replicate numpy's linalg submodule
class linalg(object):

    @staticmethod
    def svd(a, full_matrices=True, compute_uv=True, hermitian=False):

        if hermitian:
            raise NotImplementedError()

        some = not full_matrices

        U, S, V = torch.svd(a, compute_uv=compute_uv, some=some)

        return U, S, transpose(V)

    @staticmethod
    def norm(a):
        raise NotImplementedError()

    @staticmethod
    def det(a):
        raise NotImplementedError()


class random(object):

    @staticmethod
    def randint():
        raise NotImplementedError()
