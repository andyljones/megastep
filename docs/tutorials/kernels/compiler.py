import torch.utils.cpp_extension
import sysconfig

[torch_libdir] = torch.utils.cpp_extension.library_paths()
python_libdir = sysconfig.get_config_var('LIBDIR')
libpython_ver = sysconfig.get_config_var('LDVERSION')

cuda = torch.utils.cpp_extension.load(
    name='testkernels',
    sources=['wrappers.cpp'],
    extra_cflags=['-std=c++17'],
    extra_cuda_cflags=['--use_fast_math', '-lineinfo', '-std=c++14'],
    extra_ldflags=[
        f'-lpython{libpython_ver}', '-ltorch', '-ltorch_python', '-lc10_cuda', '-lc10', 
        f'-L{torch_libdir}', f'-Wl,-rpath,{torch_libdir}',
        f'-L{python_libdir}', f'-Wl,-rpath,{python_libdir}'])