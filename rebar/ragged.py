import numpy as np
import torch
import numbers

class RaggedNumpy:

    def __init__(self, vals, widths):
        self.vals = vals
        self.widths = widths
        self.starts = widths.cumsum().astype(int) - widths
        self.ends = widths.cumsum().astype(int)

        indices = np.zeros(self.widths.sum(), dtype=self.starts.dtype)
        indices[self.starts] = np.ones_like(self.starts)
        self.inverse = indices.cumsum().astype(int) - 1

    def __getitem__(self, x):
        if isinstance(x, numbers.Integral):
            return self.vals[self.starts[x]:self.ends[x]]
        if isinstance(x, slice):
            assert x.step in (None, 1)
            start, end = x.start or 0, x.stop or len(self.ends) 
            return RaggedNumpy(
                self.vals[self.starts[start]:self.ends[end-1]],
                self.widths[start:end])
        raise ValueError(f'Can\'t handle index "{x}"')

class RaggedTorch:

    def __init__(self, vals, widths):
        self.vals = vals
        self.widths = widths
        self.starts = widths.cumsum(0).long() - widths
        self.ends = widths.cumsum(0).long()

        indices = self.starts.new_zeros(self.widths.sum())
        flags = torch.ones_like(self.starts)
        self.inverse = indices.scatter(0, self.starts, flags).cumsum(0).long() - 1

    def __getitem__(self, x):
        if isinstance(x, numbers.Integral):
            return self.vals[self.starts[x]:self.ends[x]]
        if isinstance(x, slice):
            assert x.step in (None, 1)
            start, end = x.start or 0, x.stop or len(self.ends) 
            return RaggedTorch(
                self.vals[self.starts[start]:self.ends[end-1]],
                self.widths[start:end])
        raise ValueError(f'Can\'t handle index "{x}"')

def Ragged(vals, widths):
    if isinstance(vals, np.ndarray):
        return RaggedNumpy(vals, widths)
    if isinstance(vals, torch.Tensor):
        return RaggedTorch(vals, widths)
    raise ValueError(f'Can\'t create a ragged from {type(vals)}')


def test_ragged_numpy():
    vals = np.array([0, 1, 2, 3, 4, 5])
    widths = np.array([3, 1, 2])
    ragged = RaggedNumpy(vals, widths)

    np.testing.assert_allclose(ragged[1], [3])
    np.testing.assert_allclose(ragged[-1], np.array([4, 5]))

    np.testing.assert_allclose(ragged[:2].vals, [0, 1, 2, 3])
    np.testing.assert_allclose(ragged[:2].widths, [3, 1])

    np.testing.assert_allclose(ragged[1:].vals, np.array([3, 4, 5]))
    np.testing.assert_allclose(ragged[1:].widths, np.array([1, 2]))

def test_ragged_torch():
    vals = torch.as_tensor([0, 1, 2, 3, 4, 5])
    widths = torch.as_tensor([3, 1, 2])
    ragged = RaggedTorch(vals, widths)

    torch.testing.assert_allclose(ragged[1], [3])
    torch.testing.assert_allclose(ragged[-1], np.array([4, 5]))

    torch.testing.assert_allclose(ragged[:2].vals, [0, 1, 2, 3])
    torch.testing.assert_allclose(ragged[:2].widths, [3, 1])

    torch.testing.assert_allclose(ragged[1:].vals, np.array([3, 4, 5]))
    torch.testing.assert_allclose(ragged[1:].widths, np.array([1, 2]))

