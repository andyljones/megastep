.. raw:: html

    <bold><span style="color: red">This repo is still pre-release. It's public because it makes some bits of testing &
    distribution setup easier. If you come across it before this warning is removed, don't expect things to
    work reliably or consistently.</span></bold>

########
megastep
########

**megastep** helps you build 1-million FPS reinforcement learning environments *on a single GPU*.

Examples
********
**Explorer**: 180k FPS, :github:`100 lines of Python <megastep/demo/envs/explorer.py>`.

.. raw:: html

    <video controls src="_static/explorer.mp4" autoplay loop muted type="video/mp4" width=640></video>

**Deathmatch**: 1.2m FPS, :github:`150 lines of Python <megastep/demo/envs/deathmatch.py>`.

.. raw:: html

    <video controls src="_static/deathmatch.mp4" autoplay loop muted type="video/mp4" width=640></video>

Quoted FPS are for a single RTX 2080 Ti and random actions; visualizations are with a 256-neuron LSTM.

Features
********
 * Run thousands of environments in parallel, entirely on the GPU.
 * Write your own environments using PyTorch alone, no CUDA necessary.
 * 1D observations. The world is more interesting horizontally than vertically.
 * One or many agents, and one or many cameras per agent.
 * A database of 5000 home layouts to explore, based on `Cubicasa5k <https://github.com/CubiCasa/CubiCasa5k>`_.
 * A minimal, modular library. Not a framework.
 * Extensive documentation, tutorials and explanations. 

.. _install:

Install
*******
If you're working on an Ubuntu machine with CUDA already set up, it should be as simple as ::

    pip install git+https://github.com/andyljones/megastep.git#egg=megastep[cubicasa,rebar]

This installs everything needed to run the demos and tutorials. If you want :ref:`something minimal <subpackages>` ::

    pip install git+https://github.com/andyljones/megastep.git

**The first time you import megastep on your machine will be slow** as it compiles the C++ side of things.

Getting Started
***************
Having :ref:`installed megastep <install>`, you have a few choices. 

If you'd like something comprehensive, read :ref:`Writing a Minimal Environment <minimal-env>`.

If you'd like a more self-propelled introduction, read :ref:`Playing With Megastep <playing>`.

Or, you can simply browse the :github:`demos <megastep/demo>`.

Index
*****
.. toctree::
    :maxdepth: 2
    
    tutorials/index
    concepts
    faq

.. toctree::
    :maxdepth: 3

    reference

