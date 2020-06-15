import re
from functools import wraps
from pkg_resources import resource_filename
import torch.utils.cpp_extension
import torch
from rebar import dotdict, arrdict

MOVEMENTS = 7
AGENT_WIDTH = .15
TEXTURE_RES = .05
DEBUG = False

# Used for collision radius and near camera plane
AGENT_RADIUS = 1/2**.5*AGENT_WIDTH

def gamma_encode(x): 
    """Converts to viewable values"""
    return x**(1/2.2)

def gamma_decode(x):
    """Converts to interpolatable values"""
    return x**2.2

def cuda(res, supersample, fov, fps, **kwargs):
    files = [resource_filename(__package__, f'src/{fn}') for fn in ('wrappers.cpp', 'kernels.cu')]
    includes = [resource_filename(__package__, 'include')]
    cflags = ['-std=c++17'] + (['-g'] if DEBUG else [])
    cudaflags = ['--use_fast_math', '-lineinfo', '-std=c++14'] + (['-g', '-G'] if DEBUG else [])
    ldflags = [
        '-L/opt/conda/lib/python3.7/site-packages/torch/lib', '-ltorch', '-ltorch_python', '-lc10_cuda', '-lc10', 
        '-L/opt/conda/lib', '-lpython3.7m',
        '-Wl,-rpath,/opt/conda/lib/python3.7/site-packages/torch/lib',
        '-Wl,-rpath,/opt/conda/lib']
    cuda = torch.utils.cpp_extension.load('simulator', files, extra_include_paths=includes, 
                            extra_cflags=cflags, extra_cuda_cflags=cudaflags,
                            extra_ldflags=ldflags)

    cuda.initialize(float(AGENT_RADIUS), int(supersample*res), float(fov), float(fps))
    return cuda


def stack(ds):
    exemplar = ds[0]
    return exemplar.__class__([(k, [d[k] for d in ds]) for k in exemplar])

def split(ds):
    exemplar = next(iter(ds.values()))
    return [dotdict({k: v[i] for k, v in ds.items()}) for i in range(len(exemplar))]

def unpack(d):
    if isinstance(d, torch.Tensor):
        return d
    return arrdict({k: unpack(getattr(d, k)) for k in dir(d) if not k.startswith('_')})

def clone(xs):
    if isinstance(xs, dict):
        return type(xs)({k: clone(v) for k, v in xs.items()})
    return xs.clone()

# Full: 1 design: 27ms; 32 designs: 500ms 
# Empty: 1 design: 16ms; 32 designs: 120ms