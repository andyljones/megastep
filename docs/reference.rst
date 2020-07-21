.. _api:

#############
API Reference
#############

megastep
========

core
----
.. automodule:: megastep.core
    :members: Core, gamma_encode, gamma_decode
    :undoc-members:

cuda
----
.. automodule:: megastep.cuda
    :members: initialize, bake, physics, render, Render, Agents, Scenery, 
    :undoc-members:

cubicasa
--------
.. automodule:: megastep.cubicasa
    :members: sample 

geometry
--------
.. automodule:: megastep.geometry
    :members: centers, indices, display, cyclic_pairs, masks

modules
-------
.. automodule:: megastep.modules
    :members: 
    :undoc-members:

plotting
--------
.. automodule:: megastep.plotting
    :members:
    :undoc-members:

ragged
------
.. automodule:: megastep.ragged
    :members: Ragged, RaggedNumpy
    :undoc-members:

scene
-----
.. automodule:: megastep.scene
    :members: 
    :undoc-members:

spaces
------
.. automodule:: megastep.spaces
    :members:
    :undoc-members:

toys 
----
.. automodule:: megastep.toys
    :members: box, column

demo
====
.. automodule:: megastep.demo
    :members: Agent, demo, train
    :undoc-members:

heads
-----
.. automodule:: megastep.demo.heads
    :members:
    :undoc-members:

learning
--------
.. automodule:: megastep.demo.learning
    :members:
    :undoc-members:

lstm
----
.. automodule:: megastep.demo.lstm
    :members:
    :undoc-members:

demo.envs
=========

minimal
-------
.. automodule:: megastep.demo.envs.minimal
    :members: Minimal, Agent
    :undoc-members:

deathmatch
----------
.. automodule:: megastep.demo.envs.deathmatch
    :members: Deathmatch

explorer
--------
.. automodule:: megastep.demo.envs.explorer
    :members: Explorer

rebar
=====
.. automodule:: rebar

arrdict
-------
.. automodule:: rebar.arrdict
    :members: arrdict, torchify, numpyify, stack, cat

dotdict
-------
.. automodule:: rebar.dotdict
    :members: dotdict, mapping, starmapping, leaves

recording
---------
.. automodule:: rebar.recording
    :members: Encoder, ParallelEncoder
    :undoc-members:
