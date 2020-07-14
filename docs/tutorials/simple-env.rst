====================
Writing a Simple Env
====================

In this tutorial we're going to write the simplest possible environment. It's going to have one agent that gets
rewarded for colliding with a wall. Not a particularly interesting task, but it'll be easy to follow, and give us a
base of techniques that we can later build more interesting environments on top of.

TODO: Animation of collision env

Geometry
--------

The place to start is with the :ref:`geometry <geometry>`. The geometry describes the walls, rooms and lights of
an environment. In later tutorials we're going to generate thousands of unique geometries, but here for our
simplest-possible env, a single geometry will do. A single, simple geometry:

TODO: Image of box env

Yup, it's a box. Four walls and one room. To create this geometry, we start with the corners in order::

    import numpy as np
    corners = np.array([
        [0, 0]
        [0, 1]
        [1, 1]
        [1, 0]]

These corners give a 1m box, which is a bit too small for our purposes. We can scale it up by multiplying by the
width we want. It's also a good idea to shift it 1m up and to the right, as lots of machinery in megastep assumes
that everything happens in the top-right quadrant (ie, above and to the right of the origin). There's no fundamental
reason for this, it just simplifes some stuff internally.