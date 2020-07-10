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
    """Converts an array or a dict of numpy arrays to CPU tensors.

    If you'd like CUDA tensors, follow the tensor-ification ``.cuda()`` ; the attribute delegation
    built into :class:`rebar.dotdict.dotdict` s will do the rest.
    
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
    """Converts an array or a dict of tensors to numpy arrays.
    """
    if isinstance(tensors, tuple):
        return tuple(numpyify(t) for t in tensors)
    if isinstance(tensors, torch.Tensor):
        return tensors.clone().detach().cpu().numpy()
    return tensors

def stack(x, *args, **kwargs):
    """Stacks a sequence of arrays, tensors or dicts thereof.  

    For example, 

    >>> d = arrdict(a=1, b=np.array([1, 2]))
    >>> stack([d, d, d])
    arrdict:
    a    ndarray((3,), int64)
    b    ndarray((3, 2), int64)

    Any ``*args`` or ``**kwargs`` will be forwarded to the ``np.stack`` or ``torch.stack`` call. 

    Python scalars are converted to numpy scalars, so - as in the example above - stacking floats will
    get you a 1D array.
    """
    if isinstance(x[0], dict):
        ks = x[0].keys()
        return x[0].__class__({k: stack([y[k] for y in x], *args, **kwargs) for k in ks})
    if TORCH and isinstance(x[0], torch.Tensor):
        return torch.stack(x, *args, **kwargs)
    if isinstance(x[0], np.ndarray):
        return np.stack(x, *args, **kwargs) 
    if np.isscalar(x[0]):
        return np.array(x, *args, **kwargs)
    raise ValueError(f'Can\'t stack {type(x[0])}')

def cat(x, *args, **kwargs):
    """Concatenates a sequence of arrays, tensors or dicts thereof.  

    For example, 

    >>> d = arrdict(a=1, b=np.array([1, 2]))
    >>> cat([d, d, d])
    arrdict:
    a    ndarray((3,), int64)
    b    ndarray((6,), int64)

    Any ``*args`` or ``**kwargs`` will be forwarded to the ``np.concatenate`` or ``torch.cat`` call. 

    Python scalars are converted to numpy scalars, so - as in the example above - concatenating floats will
    get you a 1D array. 
    """
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
    # This is done with a factory because I am a lazy man and I didn't fancy defining all the binary ops on 
    # the arrdict manually.

    class _arrdict_base(dotdict.dotdict):
        """An arrdict is an :class:`rebar.dotdict.dotdict` with extra support for array and tensor values.

        **Indexing**

        Indexing operations are forwarded to the values:

        >>> d = arrdict(a=np.array([1, 2]), b=np.array([3, 4]))
        >>> d[0].item()  # the .item() call is needed to get it to print nicely
        arrdict:
        a    1
        b    3

        All the kinds of indexing that the underlying arrays/tensors support is supported by arrdict.

        **Binary operations**

        You can combine two arrdicts in all the ways you'd combine the underlying items

        >>> d = arrdict(a=1, b=2)
        >>> d + d
        arrdict:
        a    2
        b    4

        It works equally well with Python scalars, arrays, and tensors, and pretty much every op you're
        likely to use is covered. Call ``dir(arrdict)`` to get a list of the supported magics.
        """

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

    methods['__doc__'] = _arrdict_base.__doc__

    return type('arrdict', (_arrdict_base,), methods)

arrdict = _arrdict_factory()