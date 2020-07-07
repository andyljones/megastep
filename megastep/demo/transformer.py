import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from rebar import recurrence
import matplotlib.pyplot as plt

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, lim=1024):
        super().__init__()
        assert d_model % 2 == 0

        self._lim = lim
        inv_freq = 2*np.pi / (lim ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = pos_seq[..., None]*self.inv_freq
        return torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

    def _pattern(self, K, s):
        K = self._lim if K is None else K
        s = 1 if s is None else s
        pos_seq = torch.arange(0., K, s, dtype=torch.float, device=self.inv_freq.device)
        return self(pos_seq)

    def plot_pattern(self, K=None, s=None):
        plt.imshow(numpyify(self._pattern(K, s)))

    def plot_similarity(self, K=None, s=None):
        pattern = self._pattern(K, s)
        top = (pattern[None, :]*pattern[:, None]).sum(-1)
        bot = pattern.pow(2).sum(-1).pow(.5)
        plt.imshow(numpyify(top/(bot[None, :]*bot[:, None])), vmin=-1, vmax=+1, cmap=plt.cm.RdBu_r)
        plt.colorbar()

def rel_shift(x):
    """Explanation: https://github.com/kimiyoung/transformer-xl/issues/8#issuecomment-454458852"""
    (T, C), tail = x.shape[:2], x.shape[2:]

    zero_pad = x.new_zeros((T, 1) + tail)
    x_padded = torch.cat([zero_pad, x], dim=1)
    x_padded = x_padded.view((C+1, T) + tail)

    return x_padded[1:].view_as(x)

class ResetMasker(nn.Module):

    def __init__(self, mem_len):
        super().__init__()
        self._mem_len = mem_len
        self._reset = recurrence.State()

    @torch.no_grad()
    def __call__(self, reset):
        if reset.ndim == 1:
            return self(reset[None])
        old_reset = self._reset.get(lambda: torch.ones_like(reset[:0]))
        resets = torch.cat([old_reset, reset], 0)
        self._reset.set(resets[-self._mem_len:])

        diag_resets = torch.diag_embed(resets.T, offset=0, dim1=0, dim2=2)
        flipped = diag_resets.int().flip(2)
        rows = (flipped.cumsum(2) - flipped).flip(2)
        cols = (rows.cumsum(0)).bool()

        T = reset.size(0)
        return cols[-T:]

def attention_mask(T, M, mem_len, device=None, future=0):
    all_ones = torch.ones((T, M+T), dtype=torch.uint8, device=device)
    mask_len = M+T - mem_len
    mask_shift_len = (T - mask_len) if mask_len > 0 else T
    future_mask = torch.triu(all_ones, M+future)
    history_mask = torch.tril(all_ones, -mask_shift_len)
    default_mask = (future_mask + history_mask)[:, :, None].bool()
    return default_mask

class Weights(nn.Module):

    def __init__(self, mem_len, d_model, d_head=None, n_head=1, content=True, position=True, norm=True, memory=True):
        """Useful to be able to turn content/norm off for testing"""
        super().__init__()

        self.d_model = d_model
        self.mem_len = mem_len
        self.n_head = n_head
        self.d_head = (d_head or d_model//self.n_head)
        self.content = content
        self.position = position
        self.memory = memory

        self.m = recurrence.State()
        self.masker = ResetMasker(self.mem_len)

        self.norm = nn.LayerNorm(d_model) if norm else lambda x: x
        self.q = nn.Linear(d_model, self.n_head*self.d_head, bias=False)

        # Doesn't really matter how we initialize; LayerNorm will stop things from blowing up
        if self.content:
            self.k = nn.Linear(d_model, self.n_head*self.d_head, bias=False)
            self.k_bias = nn.Parameter(torch.empty((self.n_head, self.d_head)))
            torch.nn.init.normal_(self.k_bias, 0, 1)

        if self.position:
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r = nn.Linear(d_model, self.n_head*self.d_head, bias=False)
            self.r_bias = nn.Parameter(torch.empty((self.n_head, self.d_head)))
            torch.nn.init.normal_(self.r_bias, 0, 1)

    def forward(self, h, reset=None):
        NH, DH = self.n_head, self.d_head
        T, B = h.shape[:2]

        m = self.m.get(lambda: h.new_zeros((0, B, self.d_model)))
        if self.memory:
            self.m.set(torch.cat([m, h])[-self.mem_len:].detach())
        M = m.size(0)
        TM = T+M

        reset = torch.zeros((T, B), dtype=torch.bool, device=h.device) if reset is None else reset
        mask = self.masker(reset).permute(0, 2, 1)[:, -TM:]
        mask = mask | attention_mask(T, M, self.mem_len, device=h.device)
        mask = mask.unsqueeze(-1)

        cat = self.norm(torch.cat([m, h], 0))
        q = self.q(cat[-T:]).view(T, B, NH, DH)

        score = h.new_zeros((T, TM, B, NH))

        if self.content:
            k = self.k(cat).view(TM, B, NH, DH)
            attn_score = torch.einsum('ibnd,jbnd->ijbn', (q + self.k_bias, k))
            score += attn_score

        if self.position:
            pos_seq = torch.flip(torch.arange(TM, device=h.device, dtype=h.dtype), (0,))
            r = self.r(self.pos_emb(pos_seq)).view(TM, NH, DH)

            pos_score = torch.einsum('ibnd,jnd->ijbn', (q + self.r_bias, r))
            score += rel_shift(pos_score)

        #### compute attention probability
        score = (score
                        .div(self.d_head**.5)
                        .masked_fill(mask, -65000) # -65k is a very, very small number for a 16-bit float
                        .clamp(-65000, +65000)) # Clamp to suppress any infs that sometimes overflow when using 16-bit floats

        prob = F.softmax(score, dim=1)

        # Zero any output rows where the mask was all true
        return prob.where(~mask.all(1, keepdim=True), torch.zeros_like(prob))

class Values(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.weights = Weights(*args, **kwargs)
        self.d_model = self.weights.d_model
        self.n_head = self.weights.n_head
        self.d_head = self.weights.d_head
        self.mem_len = self.weights.mem_len
        
        self.norm = nn.LayerNorm(self.d_model)

        self.v = nn.Linear(self.d_model, self.n_head*self.d_head, bias=False)
        self.o = nn.Linear(self.n_head*self.d_head, self.d_model, bias=False)

    def forward(self, h, reset=None):
        NH, DH = self.n_head, self.d_head
        T, B = h.shape[:2]

        #TODO: Cache the qkv values rather than the h values?
        m = self.weights.m.get(lambda: h.new_zeros((0, B, self.d_model)))
        M = m.size(0)
        TM = T+M

        cat = self.norm(torch.cat([m, h], 0))
        v = self.v(cat).view(TM, B, NH, DH)

        weights = self.weights(h, reset)
        summary = torch.einsum('ijbn,jbnd->ibnd', weights, v)

        return F.relu(self.o(summary.reshape(T, B, NH*DH)))

class Gate(nn.Module):

    def __init__(self, d_model, bias=2.):
        super().__init__()
        self.W = nn.Linear(d_model, 3*d_model, bias=False)
        self.U = nn.Linear(d_model, 2*d_model, bias=False)
        self.Ug = nn.Linear(d_model, d_model, bias=False)
        self.b = nn.Parameter(torch.full((d_model,), bias))

    def forward(self, x, y):
        wr, wz, wg = torch.chunk(self.W(y), 3, dim=-1)
        ur, uz  = torch.chunk(self.U(y), 2, dim=-1)

        r = torch.sigmoid(wr + ur)
        z = torch.sigmoid(wz + uz - self.b)
        h = torch.tanh(wg + self.Ug(r*x))

        return (1 - z)*x + z*h

class GatedAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.attn = Values(**kwargs)
        self.attn_gate = Gate(self.attn.d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(self.attn.d_model),
            nn.Linear(self.attn.d_model, self.attn.d_model),
            nn.ReLU())
        self.ff_gate = Gate(self.attn.d_model)

    def forward(self, h, reset):
        a = self.attn(h, reset)
        h = self.attn_gate(h, a)
        return self.ff_gate(h, self.ff(h))

class Transformer(nn.Module):

    def __init__(self, n_layers=1, *args, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([GatedAttention(*args, **kwargs) for _ in range(n_layers)])

    def forward(self, h, reset):
        """h: TxBxC or BxC (will be treated as a single step)"""
        if h.ndim == 2:
            return self(h[None], reset[None])[0]

        for layer in self.layers:
            h = layer(h, reset)
        return h


def test_weights_simple():
    T, B, K = 5, 7, 2
    h = torch.rand((T, B, K))
    attn = Weights(T, K)
    attn(h)
    attn(h)

def test_weights_content_single():
    h = torch.tensor([10, 0, 10])[:, None, None].float()

    attn = Weights(3, 1, position=False, norm=False)
    attn.k.weight[:] = +1.
    attn.q.weight[:] = +1.
    attn.k_bias[:] = 0.
    what = attn(h)

    w = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 0]]).float()
    torch.testing.assert_allclose(w, what[:, :, 0, 0])

def test_weights_content_multi():
    h0 = torch.tensor([10, 0])[:, None, None].float()
    h1 = torch.tensor([10,])[:, None, None].float()

    attn = Weights(3, 1, position=False, norm=False)
    attn.k.weight[:] = +1.
    attn.q.weight[:] = +1.
    attn.k_bias[:] = 0.

    attn(h0)
    what = attn(h1)

    w = torch.tensor([[1, 0, 0]]).float()
    torch.testing.assert_allclose(w, what[:, :, 0, 0])

def test_values_simple():
    T, B, K = 5, 7, 2
    h = torch.rand((T, B, K))
    attn = Values(T, K)
    attn(h)
    attn(h)

def test_reset():
    h = torch.rand((8, 1, 2))
    reset = torch.zeros((8, 1), dtype=torch.bool)
    reset[6] = True

    weights = Weights(4, 2)
    weights(h, reset)
    attn = weights(h, reset)

    assert (attn[0:, :2] == 0).all()
    assert (attn[6:, :10] == 0).all()