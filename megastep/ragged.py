import numpy as np
import torch
import numbers
from rebar import arrdict
from . import cuda

class RaggedNumpy:

    def __init__(self, vals, widths):
        """A :ref:`Ragged <raggeds>` backed by numpy arrays.

        :param vals: an array or tensor of values
        :param widths: an array or tensor of widths of each element in the ragged array.
        """
        self.vals = vals
        self.widths = widths
        self.starts = widths.cumsum().astype(int) - widths
        self.ends = widths.cumsum().astype(int)

        assert widths.sum() == vals.shape[0]

        indices = np.zeros(self.widths.sum(), dtype=self.starts.dtype)
        indices[self.starts] = np.ones_like(self.starts)
        self.inverse = indices.cumsum().astype(int) - 1

    def __getitem__(self, x):
        """Indexes or slices the first dimension of the ragged array."""
        if isinstance(x, numbers.Integral):
            return self.vals[self.starts[x]:self.ends[x]]
        if isinstance(x, slice):
            assert x.step in (None, 1)
            start, end = x.start or 0, x.stop or len(self.ends) 
            return RaggedNumpy(
                self.vals[self.starts[start]:self.ends[end-1]],
                self.widths[start:end])
        raise ValueError(f'Can\'t handle index "{x}"')

    def torchify(self):
        """Applies :func:`arrdict.torchify` to the backing arrays and returns a new :class:`cuda.Ragged$ND` for them"""
        return Ragged(
            arrdict.torchify(self.vals),
            arrdict.torchify(self.widths))

def Ragged(vals, widths):
    """Returns a :ref:`Ragged <raggeds>` array or tensor. 
    
    If you pass numpy arrays as arguments, you'll get back a :class:`RaggedNumpy` object; if you pass Torch tensors,
    you'll get back a :class:`cuda.Ragged$ND` that's backed by a C++
    implementation and is OK to pass to the :class:`core.Core` machinery.

    :param vals: an array or tensor of values
    :param widths: an array or tensor of widths of each element in the ragged array.
    """
    if isinstance(vals, np.ndarray):
        return RaggedNumpy(vals, widths)
    Ragged = getattr(cuda, f'Ragged{vals.ndim}D')
    return Ragged(vals, widths)

def test_ragged():
    vals = torch.as_tensor([0, 1, 2, 3, 4, 5]).float()
    widths = torch.as_tensor([3, 1, 2]).int()
    ragged = Ragged(vals, widths)

    torch.testing.assert_allclose(ragged[1], [3])
    torch.testing.assert_allclose(ragged[-1], [4, 5])

    torch.testing.assert_allclose(ragged[:2].vals, [0, 1, 2, 3])
    torch.testing.assert_allclose(ragged[:2].widths, [3, 1])

    torch.testing.assert_allclose(ragged[1:].vals, np.array([3, 4, 5]))
    torch.testing.assert_allclose(ragged[1:].widths, np.array([1, 2]))

def test_ragged_numpy():
    vals = np.array([0, 1, 2, 3, 4, 5])
    widths = np.array([3, 1, 2])
    ragged = RaggedNumpy(vals, widths)

    np.testing.assert_allclose(ragged[1], [3])
    np.testing.assert_allclose(ragged[-1], [4, 5])

    np.testing.assert_allclose(ragged[:2].vals, [0, 1, 2, 3])
    np.testing.assert_allclose(ragged[:2].widths, [3, 1])

    np.testing.assert_allclose(ragged[1:].vals, np.array([3, 4, 5]))
    np.testing.assert_allclose(ragged[1:].widths, np.array([1, 2]))
