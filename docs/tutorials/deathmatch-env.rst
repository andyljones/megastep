.. _deathmatch-env:

================
A Deathmatch Env
================

In this tutorial we're going to take what we learned in the :ref:`explorer env tutorial <explorer-env>` and build a
the :class:`~megastep.demo.envs.deathmatch.Deathmatch` environment. 

.. raw:: html

    <video controls src="../_static/deathmatch.mp4" autoplay loop muted type="video/mp4" width=640></video>

Compared to the explorer env, the deathmatch env is a competitive, multiagent environment with a very different reward
function.

You can see the final deathmatch code :github:`here <megastep/demo/envs/deathmatch.py>`.

Like in the second half of the minimal environment tutorial, when discussing the construction of a large class it
isn't feasible to copy-paste the entire definition every time. Instead, the snippets here are going to be small 
'experiments' that demonstrate how a particular part of the class works.

TODO-DOCS Rest of the deathmatch tutorial
