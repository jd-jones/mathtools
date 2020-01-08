def set_backend(name):
    """

    Parameters
    ----------
    name : str in {'numpy', 'torch'}
    """

    global np

    if name == 'numpy':
        import numpy as np
    elif name == 'torch':
        from mathtools import torch_wrapper as np
    else:
        err_str = f""
        raise AssertionError(err_str)


def set_default_device(device):
    """

    Parameters
    ----------
    device : torch.device
    """

    np.DEFAULT_DEVICE = device

    np.pi = np.pi.to(device=device)
    np.inf = np.inf.to(device=device)


set_backend('numpy')
