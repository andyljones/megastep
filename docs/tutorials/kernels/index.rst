.. _tutorial-kernels:

==============
Custom Kernels
==============
In most any program you care to write, a small part of the code will make up the overwhelming majority of the runtime.
The idea behind megastep is that you can write *almost* all of your environment in PyTorch, and then write the small,
majority-of-the-runtime bit in CUDA. 

While megastep's :func:`~megastep.cuda.render` and :func:`~megastep.cuda.physics` calls make up the slow bits of the 
environments I've been prone to write, it's not likely they cover all of your use-cases. In fact, if you're reading 
this it probably means you've decided that they *don't* cover your use cases. So this tutorial is about writing your 
own.

There is not much in this tutorial that isn't in `the official PyTorch extension tutorial
<https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-the-c-op>`_. If you find yourself confused about 
something written here, you can get another perspective on it there. However that tutorial spends a lot of time 
discussing things like gradients that aren't as interesting to us.

At a high level, this tutorial is first going to discuss compiling C++ into a Python module. Then we're going to
talk about using C++ to do PyTorch computations, and then we're going to discuss using CUDA to do PyTorch computations. 

Prerequesites
*************
TODO-DOCS Explain the prerequesites

Turning C++ into Python
***********************
The start of our journey is with a C++ file, which we're going to call ``wrappers.cpp`` for reasons that will become
clear later. Make yourself a ``wrappers.cpp`` file in your working directory with the following strange incantations:

.. code-block:: cpp

    #include <torch/extension.h>

    int addone(int x) { return x + 1; }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("addone", &addone)
    }

Let's work through this.
 * ``#include``: This header pulls in a lot of PyTorch's `C++ API <https://pytorch.org/cppdocs/>`_,
   but more importantly it pulls in `pybind <https://pybind11.readthedocs.io/en/stable/intro.html>`_. 
   Pybind is, in a word, magic. It lets you package C++ code up into Python modules, and goes a long way to automating
   the conversion of Python objects into C++ types and vice versa.
 * ``addone``: Next we define a function that we'd like to call from Python.
 * ``PYBIND11_MODULE``: Then we invoke `pybind's module creation macro <https://pybind11.readthedocs.io/en/master/reference.html?highlight=PYBIND11_MODULE#c.PYBIND11_MODULE>`_.
   It takes the name of the module (``TORCH_EXTENSION_NAME``, which evaluates to a specific torch-provided name) and which provides
   a variable - ``m`` - that'll be used to identify which bits of C++ need to be hooked up to Python.
 * ``m.def``: Finally, we specify the address of the thing we want to call from Python - ``&addone`` - and we
   give specify the name that thing should be known by on the Python side - ``"addone"``.

Now, the Python side. Make an ``wrapper.py`` file in the same directory containing ::

    import torch.utils.cpp_extension
    import sysconfig

    [torch_libdir] = torch.utils.cpp_extension.library_paths()
    python_libdir = sysconfig.get_config_var('LIBDIR')
    libpython_ver = sysconfig.get_config_var('LDVERSION')

    cuda = torch.utils.cpp_extension.load(
        name='testkernels',
        sources=['wrappers.cpp'],
        extra_cflags=['-std=c++17'],
        extra_cuda_cflags=['-std=c++14', '-lineinfo'],
        extra_ldflags=[
            f'-lpython{libpython_ver}', '-ltorch', '-ltorch_python', '-lc10_cuda', '-lc10', 
            f'-L{torch_libdir}', f'-Wl,-rpath,{torch_libdir}',
            f'-L{python_libdir}', f'-Wl,-rpath,{python_libdir}'])

Almost all of this is boilerplate C++ compilation voodoo; the only really important bits to note are the name - which is 
what our new C++ module will be added to the import system under - and the list of source files. I explain the rest of 
the options :ref:`below <switches>` if you're interested, but frankly you can skip reading it until such time as compilation is 
giving you trouble.

With this file defined, we can test things out! Find yourself a terminal and run ::

    python -c "from wrapper import *; print(cuda.addone(1))"

It should hang for a while while it compiles in the background, then print 2! If it does, congrats - you're sending data 
over to Python, doing some computation, and getting it back! 

If for some reason it *doesn't* work, the first thing to do is to add a ``verbose=True`` arg to the ``load()`` call. 
That'll give you much more detailed debugging information, and hopefully let you ID the problem. 

As an aside, while I usually work from a notebook but when doing lots of Python/C++ developmet I prefer to invoke
things from a terminal. In a notebook, a failed compilation can sometimes be silently 'covered' by torch
loading an old version of your module, and that way madness lies.

.. _switches:

Compilation Switches
--------------------
To save some scrolling, here's the compilation snippet from earlier::

    import torch.utils.cpp_extension
    import sysconfig

    [torch_libdir] = torch.utils.cpp_extension.library_paths()
    python_libdir = sysconfig.get_config_var('LIBDIR')
    libpython_ver = sysconfig.get_config_var('LDVERSION')

    cuda = torch.utils.cpp_extension.load(
        name='testkernels',
        sources=['wrappers.cpp'],
        extra_cflags=['-std=c++17'],
        extra_cuda_cflags=['-std=c++14', '-lineinfo'],
        extra_ldflags=[
            f'-lpython{libpython_ver}', '-ltorch', '-ltorch_python', '-lc10_cuda', '-lc10', 
            f'-L{torch_libdir}', f'-Wl,-rpath,{torch_libdir}',
            f'-L{python_libdir}', f'-Wl,-rpath,{python_libdir}'])

And the notes:
 * ``[torch_libdir]``: Find the path to the directory of Torch C++ libraries we need to link against.
 * ``python_libdir``: Find the path to the directory of Python C libraries we need to link against. 
 * ``libpython_ver``: We specifically want the Python C library corresponding to the version of Python we're running right now.
 * ``cuda = torch``: We're going to get torch to compile our C++ code for us, link it against a bunch of libraries and then 
   stuff it into the ``cuda`` variable.   
 * ``name='testkernels``: Our library is going to be loaded into Python as the 'testkernels' library. That is, as well as 
   it being the ``cuda`` variable, we can also access our C++ code through ``import testkernels``. 
 * ``sources``: This is the list of files to compile; in our case, just our ``wrappers.cpp``.
 * ``extra_cflags``: Here we say we want the C++ side of things compiled as C++17 code. C++ has come a long way in the last few
   years, and compiling a modern version makes for a much more pleasant time writing C++.
 * ``extra_cuda_cflags``: And here we say we want the CUDA side of things compiled as C++14 code. Not quite as nice as C++17 code,
   but the best the CUDA compiler could support as of the time I wrote this. We also chuck in the ``-lineinfo`` switch, which 
   will give us more useful debugging information when things go wrong. 
 * ``extra_ldflags``: And finally, we list off all the libraries that need to be included when linking the compiled code.

TODO-DOCS finish the kernels tutorial