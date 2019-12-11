import torch

from torch import (
    zeros, zeros_like, ones, ones_like,
    arange, meshgrid,
    eye, diag,
    any, all, sum,
    cos, sin, arctan2, radians,
    floor, ceil
)

# TODO: diff, cross, dot, logical_not, append


def array(obj):
    return torch.tensor(obj)


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
