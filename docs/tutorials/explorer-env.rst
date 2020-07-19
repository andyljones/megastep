.. _explorer-env:

===============
An Explorer Env
===============

In this tutorial we're going to take :ref:`the minimal environment we studied earlier <minimal-env>` and build it into
the :class:`~megastep.demo.envs.explorer.Explorer` environment. 

.. raw:: html

    <video controls src="../_static/explorer.mp4" autoplay loop muted type="video/mp4" width=640></video>

Compared to the minimal env, the explorer env
 * Has compound RGB-depth-`IMU <https://en.wikipedia.org/wiki/Inertial_measurement_unit>`_ observations.
 * Has movement with momentum.
 * Outputs rewards for exploring.
 * Resets agents when they've lived too long.

You can see the final explorer code :github:`here <megastep/demo/envs/explorer.py>`.

Like in the second half of the minimal environment tutorial, when discussing the construction of a large class it
isn't feasible to copy-paste the entire definition every time. Instead, the snippets here are going to be small 
'experiments' that demonstrate how a particular part of the class works.

Getting Started
***************
The first thing to do is to make sure you can actually run the complete explorer environment. Open up a notebook or
IPython console and

TODO-DOCS Rest of the explorer tutorial