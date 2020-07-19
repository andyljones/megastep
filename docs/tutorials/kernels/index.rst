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
 * First we include torch's extension header. This header pulls in a lot of PyTorch's `C++ API <https://pytorch.org/cppdocs/>`_,
   but more importantly it pulls in `pybind <https://pybind11.readthedocs.io/en/stable/intro.html>`_. 
   Pybind is, in a word, magic. It lets you package C++ code up into Python modules, and goes a long way to automating
   the conversion of Python objects into C++ types and vice versa.
 * Next, we define a function - ``addone`` - that we'd like to call from Python.
 * Then we invoke `pybind's module creation macro <https://pybind11.readthedocs.io/en/master/reference.html?highlight=PYBIND11_MODULE#c.PYBIND11_MODULE>`_,
   which takes the name of the module (``TORCH_EXTENSION_NAME``, again for reasons that will become clear soon) and which provides
   a variable - ``m`` - that'll be used to identify which bits of C++ need to be hooked up to Python.
 * Finally, we call ``m.def``. We give it the address of the thing we want to call from Python - ``&addone`` - and we
   give it the name that thing should be known by on the Python side - ``"addone"``.