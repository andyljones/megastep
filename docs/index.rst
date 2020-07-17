.. raw:: html

    <bold><span style="color: red">This repo is still pre-release. It's public because it makes some bits of testing &
    distribution setup easier. If you come across it before this warning is removed, don't expect things to
    work reliably or consistently.</span></bold>

########
megastep
########

**megastep** helps you build 1-million FPS reinforcement learning environments.

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
 * 1D observations. The world is much more interesting horizontally than vertically.
 * One or many agents, and one or many cameras per agent.
 * A database of 5000 home layouts to explore, based on `Cubicasa5k <https://github.com/CubiCasa/CubiCasa5k>`_.
 * A minimal, modular library. Not a framework.
 * Extensive documentation, tutorials and explanations. 

.. _install:

Install
*******
If you're working on an Ubuntu machine with CUDA already set up, it should be as simple as ::

    pip install git+https://github.com/andyljones/megastep.git#egg=megastep[cubicasa,rebar]

This installs everything needed to run the demos and tutorials. If you want :ref:` something minimal <subpackages>` ::

    pip install git+https://github.com/andyljones/megastep.git#egg=megastep[cubicasa,rebar]

There are some fairly hefty dependencies so it might take a while to install, but it should get there fine. If it
doesn't, my total guess is that it's the Pytorch install that will be the problem, in which case I refer you to
`their install page <https://pytorch.org/get-started/locally/>`_. 

**The first time you import megastep will be slow** as it compiles the C++ side of things.

**If you haven't got CUDA, megastep will not work**. There are some parts of megastep - like the cubicasa package - 
that you may still find useful, but in that case I recommend just copy-pasting the code you want from Github.

If you're on a different OS, then it's possible megastep will work, but I can't provide you any support. You're welcome
to ask for help on the GitHub issues page, but you'll be relying on the community to come up with an answer.

Getting Started
***************
Read the :ref:`setup instructions <install>` then 
 * If you'd like something comprehensive, read :ref:`Writing a Minimal Environment <minimal-env>`.
 * (Not yet written) If you'd like a lightweight introduction, read :ref:`Playing With Megastep <playing>`.
 * Or, you can just take a look at the :github:`demos <megastep/demo>`.

Links
*****
 * `Github <https://github.com/andyljones/megastep>`_
 * `Author <https://www.andyljones.com>`_

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

