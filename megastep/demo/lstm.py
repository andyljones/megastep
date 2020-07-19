"""TODO-DOCS LSTM docs"""
import torch
from torch import nn
from rebar import recurrence, dotdict
from torch.nn.utils.rnn import PackedSequence

class Packer:

    def __init__(self, reset):
        self._reset = reset
        ext = reset.clone()
        ext[0] = True
        ext = ext.T.flatten()

        idxs = ext.nonzero().squeeze(-1)

        b = ext.cumsum(0)-1
        t = torch.arange(len(ext), device=idxs.device) - idxs[b]

        ends = torch.cat([idxs, torch.full_like(idxs[:1], len(b))])
        l = (ends[1:] - ends[:-1])

        T = reset.size(0)
        B = len(idxs)
        L = T+1
        assert T**2 * B < 2**32 - 1

        self._b = b
        self._t = t
        self._order = torch.argsort(t*B*L + b*L + (L-l[b]-1))
        self._sizes = torch.flip(torch.flip(torch.histc(l-1, l.max(), 0, l.max()), (0,)).cumsum(0), (0,))

    def pack_data(self, x):
        vals = x.transpose(0, 1).reshape(-1, *x.shape[2:])[self._order]
        return PackedSequence(vals, self._sizes.cpu())

    def pack_state(self, h):
        T, B = self._reset.shape
        J = self._sizes[0]
        mask = (self._order[:J] % T == 0) & ~self._reset.T.flatten()[self._order[:J]]
        hp = h.new_zeros((1, self._sizes[0], *h.shape[2:]))
        hp[:, mask] = h[:, ~self._reset[0]]
        return hp

    def pack(self, x, h, c):
        return self.pack_data(x), (self.pack_state(h), self.pack_state(c))

    def unpack_data(self, xp):
        T, B = self._reset.shape
        x = xp.data.new_zeros(T*B, *xp.data.shape[1:])
        x[self._order] = xp.data
        return x.reshape(B, T, *xp.data.shape[1:]).transpose(0, 1)

    def unpack_state(self, hp):
        T, B = self._reset.shape
        mask = (self._order % T == T-1)
        left_idxs = (self._order//T)[mask]
        right_idxs = self._b[self._order][mask]

        h = hp.new_zeros((1, B, *hp.shape[2:]))
        h[:, left_idxs] = hp[:, right_idxs]
        return h

    def unpack(self, xp, hcp):
        hp, cp = hcp
        return self.unpack_data(xp), (self.unpack_state(hp), self.unpack_state(cp))


class LSTM(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        self._d_model = d_model
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model)
        self._h = recurrence.State()
        self._c = recurrence.State()

    def forward(self, x, reset):
        if x.ndim == 2:
            return self.forward(x[None], reset[None])[0]

        T, B, C = x.shape
        h0 = self._h.get(lambda: x.new_zeros(1, B, self._d_model))
        c0 = self._c.get(lambda: x.new_zeros(1, B, self._d_model))

        packer = Packer(reset)

        output = self.lstm(*packer.pack(x, h0, c0))
        y, (hn, cn) = packer.unpack(*output)
        self._h.set(hn.detach())
        self._c.set(cn.detach())

        return y


def test_packer():
    reset = torch.Tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]).bool().cuda()

    x = torch.Tensor([
        [0, 10, 20, 30],
        [1, 11, 40, 31],
        [2, 12, 41, 50]]).float().cuda()

    h = torch.Tensor([[1, 2, 3, 4]]).float().cuda()

    packer = Packer(reset)
    xp, (hp, cp) = packer.pack(x, h, h)
    xu, (hu, cu) = packer.unpack(xp, (hp, cp))

    torch.testing.assert_allclose(xp.data, torch.tensor([0, 10, 20, 40, 30, 50, 1, 11, 41, 31, 2, 12]).float().cuda())
    torch.testing.assert_allclose(xp.batch_sizes, torch.tensor([6, 4, 2]))
    torch.testing.assert_allclose(x, xu)

    torch.testing.assert_allclose(hp, torch.tensor([1, 0, 3, 0, 4, 0]).float().cuda())
    torch.testing.assert_allclose(hu, torch.tensor([[1, 0, 0, 0]]).float().cuda())

