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


set_backend('numpy')
