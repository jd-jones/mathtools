import builtins
import sys
import functools

import numpy as np
import torch
from torch import (
    arange, stack, meshgrid,
    diag, einsum, sum, unique,
    cos, sin, cross, dot, log, exp,
    floor, ceil, isinf, allclose, sign,
    # logical_not, logical_xor,
    bool, uint8, int8, short, int, long, half, float, double
)

import kornia


THIS_MODULE = sys.modules[__name__]


DEFAULT_DEVICE = None


inf = torch.tensor(np.inf)
pi = torch.tensor(np.pi)

TYPES_BY_RANK = (bool, int8, short, int, long, half, float, double)
TYPES_TO_RANK = {t: r for r, t in enumerate(TYPES_BY_RANK)}


# -=( DECORATORS )==-----------------------------------------------------------
def override_default_device(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if 'device' not in kwargs:
            kwargs['device'] = DEFAULT_DEVICE
            return func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapped


def ensure_tensor(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            args = tuple(array(a) for a in args)
            return func(*args, **kwargs)
    return wrapped


def ensure_tensor_tup(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            args = tuple(tuple(array(e) for e in a) for a in args)
            return func(*args, **kwargs)
    return wrapped


def coerce_types_tup(func):
    @functools.wraps(func)
    def wrapped(tup_arg, **kwargs):
        types = tuple(a.dtype for a in tup_arg)
        if builtins.all(t == types[0] for t in types):
            return func(tup_arg, **kwargs)
        max_rank = max(TYPES_TO_RANK[t] for t in types)
        max_type = TYPES_BY_RANK[max_rank]
        tup_arg = tuple(a.to(max_type) for a in tup_arg)
        return func(tup_arg, **kwargs)
    return wrapped


def remove_empty_tup(func):
    @functools.wraps(func)
    def wrapped(tup_arg, **kwargs):
        tup_arg = filter(len, tup_arg)
        return func(tup_arg, **kwargs)
    return wrapped


# -=( WRAPPER FUNCTIONS )==----------------------------------------------------
def diag_indices_from(x):
    if len(x.shape) != 2:
        err_str = f"Input is not a 2D array --- has shape {x.shape}"
        raise AssertionError(err_str)
    if x.shape[0] != x.shape[1]:
        err_str = f"Matrix is not square --- has shape {x.shape}"
        raise AssertionError(err_str)

    return (torch.arange(x.shape[0]),) * 2


def nonzero(x):
    nonzero_array = torch.nonzero(x)
    return tuple(col for col in transpose(nonzero_array))


def flip(x, axis=None):
    return torch.flip(x, [axis])


def any(x):
    return x.byte().any()


def all(x):
    return x.byte().all()


def logical_not(x):
    return ~(x.byte())


def logical_and(x, y):
    return x.byte() & y.byte()


def logical_or(x, y):
    return x.byte() | y.byte()


def logical_xor(x, y):
    return x.byte() ^ y.byte()


@override_default_device
def array(obj, **kwargs):
    return torch.tensor(obj, **kwargs)


@override_default_device
def zeros(*args, **kwargs):
    return torch.zeros(*args, **kwargs)


@override_default_device
@ensure_tensor
def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs)


@override_default_device
def ones(*args, **kwargs):
    return torch.ones(*args, **kwargs)


@override_default_device
@ensure_tensor
def ones_like(*args, **kwargs):
    return torch.ones_like(*args, **kwargs)


@override_default_device
def eye(*args, **kwargs):
    return torch.eye(*args, **kwargs)


def loadtxt(fn, **kwargs):
    a = np.loadtxt(fn, **kwargs)
    return array(a)


def rint(x):
    return torch.round(x)


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


@coerce_types_tup
@ensure_tensor_tup
@remove_empty_tup
def hstack(arrays):
    arrays = tuple(a[:, None] if len(a.shape) == 1 else a for a in arrays)
    return torch.cat(arrays, dim=1)


@coerce_types_tup
@ensure_tensor_tup
@remove_empty_tup
def vstack(arrays):
    arrays = tuple(a[None, :] if len(a.shape) == 1 else a for a in arrays)
    return torch.cat(arrays, dim=0)


@coerce_types_tup
@ensure_tensor_tup
@remove_empty_tup
def dstack(arrays):
    def expand(array):
        if len(array.shape) == 2:
            return array[:, :, None]
        else:
            raise AssertionError(f"array has shape {array.shape}")
    arrays = tuple(expand(a) for a in arrays)
    return torch.cat(arrays, dim=2)


@coerce_types_tup
@ensure_tensor_tup
def column_stack(arrays):
    return torch.stack(arrays, dim=1)


@coerce_types_tup
@ensure_tensor_tup
def row_stack(arrays):
    return torch.stack(arrays, dim=0)


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


# -=( NUMPY SUBMODULES )==-----------------------------------------------------
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
