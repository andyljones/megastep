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

        idxs = ext.nonzero().squeeze()

        b = ext.cumsum(0)-1
        t = torch.arange(len(ext), device=idxs.device) - idxs[b]

        ends = torch.cat([idxs, torch.full_like(idxs[:1], len(b))])
        l = (ends[1:] - ends[:-1])

        T, B = reset.shape
        assert T**2 * B < 2**32 - 1

        self._order = torch.argsort(t*B*T + b*T + (T-l[b]))
        self._sizes = torch.flip(torch.flip(torch.histc(l-1, l.max(), 0, l.max()), (0,)).cumsum(0), (0,))

    def pack_input(self, x):
        vals = x.reshape(-1, *x.shape[2:])[self._order]
        return PackedSequence(vals, self._sizes.cpu())

    def pack_state(self, h):
        T, B = self._reset.shape
        initial = self._order[::T][~self._reset[0]]
        hp = h.new_zeros((1, self._sizes[0], self._d_model))
        hp[initial] = h[~self._reset[0]]
        return hp

    def pack(self, x, h, c):
        return self.pack_input(x), (self.pack_state(h), self.pack_state(c))

    def unpack_output(self, y):
        T, B = self._reset.shape

        u = y.new_zeros(T*B, *y.shape[1:])
        u[self._order] = y

        return u.reshape(T, B)

    def unpack_state(self, h):
        T, B = self._reset.shape
        final = self._order[T-1::T]
        return h[final]

    def unpack(self, y, hc):
        h, c = hc
        return self.unpack_output(y), (self.unpack_state(h), self.unpack_state(c))

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

        return y.data[:T*B].reshape(T, B, self._d_model)



