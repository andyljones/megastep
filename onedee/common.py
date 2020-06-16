import re
from functools import wraps
from pkg_resources import resource_filename
import torch.utils.cpp_extension
import torch
from rebar import dotdict, arrdict

AGENT_WIDTH = .15
TEXTURE_RES = .05

# Used for collision radius and near camera plane
AGENT_RADIUS = 1/2**.5*AGENT_WIDTH

DEBUG = False

def gamma_encode(x): 
    """Converts to viewable values"""
    return x**(1/2.2)

def gamma_decode(x):
    """Converts to interpolatable values"""
    return x**2.2

def cuda(**kwargs):
    files = [resource_filename(__package__, f'src/{fn}') for fn in ('wrappers.cpp', 'kernels.cu')]
    includes = [resource_filename(__package__, 'include')]
    cflags = ['-std=c++17'] + (['-g'] if DEBUG else [])
    cudaflags = ['--use_fast_math', '-lineinfo', '-std=c++14'] + (['-g', '-G'] if DEBUG else [])
    ldflags = [
        '-L/opt/conda/lib/python3.7/site-packages/torch/lib', '-ltorch', '-ltorch_python', '-lc10_cuda', '-lc10', 
        '-L/opt/conda/lib', '-lpython3.7m',
        '-Wl,-rpath,/opt/conda/lib/python3.7/site-packages/torch/lib',
        '-Wl,-rpath,/opt/conda/lib']
    return torch.utils.cpp_extension.load('simulator', files, extra_include_paths=includes, 
                            extra_cflags=cflags, extra_cuda_cflags=cudaflags,
                            extra_ldflags=ldflags)