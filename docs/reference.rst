#################
Reference
#################

megastep
========

cubicasa
--------
.. automodule:: megastep.cubicasa
    :members: sample 

.. automodule:: megastep.geometry
    :members: centers, indices, display

toys 
----
.. automodule:: megastep.toys
    :members: box, column

ragged
------
.. automodule:: megastep.ragged
    :members: Ragged, RaggedNumpy
    :undoc-members:

core
----
.. automodule:: megastep.core
    :members: Core, gamma_encode, gamma_decode
    :undoc-members:

cuda
----
.. automodule:: megastep.cuda
    :members: initialize, bake, physics, render, Agents, Scene, 
    :undoc-members:

rebar
=====
**rebar** helps with reinforcement learning. It's a toolkit that helps with some common pain-points in reinforcement
learning development. Only two pieces of rebar are currently used in megastep: :mod:`rebar.dotdict` and
:mod:`rebar.arrdict`.

dotdict
-------------
.. automodule:: rebar.dotdict
    :members: dotdict, mapping, starmapping, leaves

arrdict
-------------
.. automodule:: rebar.arrdict
    :members: arrdict, torchify, numpyify, stack, cat
