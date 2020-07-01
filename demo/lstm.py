import torch
from torch import nn
from rebar import recurrence
from torch.nn.utils.rnn import PackedSequence

def pack(x, reset):
    ext = reset.clone()
    ext[0] = True
    ext = ext.T.flatten()

    idxs = ext.nonzero().squeeze()

    b = ext.cumsum(0)-1
    t = torch.arange(len(ext), device=idxs.device) - idxs[b]

    ends = torch.cat([idxs, torch.full_like(idxs[:1], len(idxs))])
    l = (ends[1:] - ends[:-1])[b]

    T, B = reset.shape
    assert T**2 * B < 2**32 - 1

    order = torch.argsort(t*B*T + b*T + (T-l))

    vals = x.reshape(-1, *x.shape[2:])[order]
    sizes = torch.flip(torch.flip(torch.histc(l, l.max()+1), (0,)).cumsum(0), (0,))

    return PackedSequence(vals, sizes.cpu())

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

        p = pack(x, reset)

        J = p.batch_sizes[0]
        hp = h0.new_zeros((1, J, self._d_model))
        hp[:, :B] = h0
        cp = h0.new_zeros((1, J, self._d_model))
        cp[:, :B] = c0

        y, (hn, cn) = self.lstm(p, (h0, c0))
        self._h.set(hn[:, :B].detach())
        self._c.set(cn[:, :B].detach())

        return y 



