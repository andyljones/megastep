from multiprocessing import Value
import numpy as np
from functools import partialmethod
from . import dotdict
try:
    import torch
    TORCH = True
except ModuleNotFoundError:
    TORCH = False

@dotdict.mapping
def torchify(a):
    """Converts an array or a dotdict of numpy arrays to CPU tensors.

    If you'd like CUDA tensors, follow the tensor-ification ``.cuda()`` ; the attribute delegation
    built into :func:`dotdict.dotdict` s will do the rest.
    
    Floats get mapped to 32-bit PyTorch floats; ints get mapped to 32-bit PyTorch ints. This is usually what you want in 
    machine learning work.
    """
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating):
        dtype = torch.float
    elif np.issubdtype(a.dtype, np.integer):
        dtype = torch.int
    elif np.issubdtype(a.dtype, np.bool_):
        dtype = torch.bool
    else:
        raise ValueError('Can\'t handle {a.dtype}')
    return torch.as_tensor(np.array(a), dtype=dtype)

@dotdict.mapping
def numpyify(tensors):
    """Converts an array or a dotdict of tensors to numpy arrays.
    """
    if isinstance(tensors, tuple):
        return tuple(numpyify(t) for t in tensors)
    if isinstance(tensors, torch.Tensor):
        return tensors.clone().detach().cpu().numpy()
    return tensors

def stack(x, *args, **kwargs):
    if isinstance(x[0], dict):
        ks = x[0].keys()
        return x[0].__class__({k: stack([y[k] for y in x], *args, **kwargs) for k in ks})
    if TORCH and isinstance(x[0], torch.Tensor):
        return torch.stack(x, *args, **kwargs)
    if isinstance(x[0], np.ndarray):
        return np.stack(x, *args, **kwargs) 
    if np.isscalar(x[0]):
        return np.array(x, *args, **kwargs)
    if isinstance(x[0], np.random.mtrand.RandomState):
        return x
    raise ValueError(f'Can\'t stack {type(x[0])}')

def cat(x, *args, **kwargs):
    if isinstance(x[0], dict):
        ks = x[0].keys()
        return x[0].__class__({k: cat([y[k] for y in x], *args, **kwargs) for k in ks})
    if TORCH and isinstance(x[0], torch.Tensor):
        return torch.cat(x, *args, **kwargs)
    if isinstance(x[0], np.ndarray):
        return np.concatenate(x, *args, **kwargs) 
    if np.isscalar(x[0]):
        return np.array(x)
    raise ValueError(f'Can\'t cat {type(x[0])}')

def _arrdict_factory():

    class _arrdict_base(dotdict.dotdict):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getitem__(self, x):
            if isinstance(x, str):
                return super().__getitem__(x)
            return type(self)({k: v[x] for k, v in self.items()})

        def __binary_op__(self, name, rhs):
            if isinstance(rhs, dict):
                return self.starmap(name, rhs)
            else:
                return super().__getattr__(name)(rhs)

    # Add binary methods
    # https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
    binaries = [
        'lt', 'le', 'eq', 'ne', 'ge', 'gt', 
        'add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod', 'divmod', 'pow', 'lshift', 'rshift', 'and', 'or', 'xor',
        'radd', 'rsub', 'rmul', 'rmatmul', 'rtruediv', 'rfloordiv', 'rmod', 'rdivmod', 'rpow', 'rand', 'lshift', 'rshift', 'ror', 'rxor']
    methods = {f'__{name}__': partialmethod(_arrdict_base.__binary_op__, f'__{name}__') for name in binaries}

    return type('arrdict', (_arrdict_base,), methods)

arrdict = _arrdict_factory()