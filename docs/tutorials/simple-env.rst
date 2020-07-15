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
reason for this, it just simplifes some stuff internally. ::

    corners = 5*corners + 1

Then to get the walls, we take all sequential pairs of corners and stack them::

    from megastep import geometry
    walls = np.stack(geometry.cyclic_pairs(corners))

You can check that these walls are what we think they are using matplotlib::

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    lines = mpl.collections.LineCollection(walls, color='k', width=2)
    plt.axes().add_collection(lines)

TODO: Image of walls

With the walls in place, the other thing to deal with is rooms. There's no strict definition of a room; they're 
just small, generic regions. The usual use of them is to make it easy to spawn multiple agents near eachother.

In this case, our room is going to just be the corners we had before. That's a list of corners though, while our 
geometry wants a mask. Fortunately there's already a function to turn one into the other::

    masks = geometry.masks(walls, [corners])

Again, we can plot it to check how it looks:

    plt.imshow(masks)

This ``masks`` array has a -1 where there's a wall, a 0 where there's free space, and a 1 where our room is. Now that
we've got both walls and masks, we can create our geometry instance and have a look at it:

    from rebar import dotdict
    g = dotdict.dotdict(
        id="box",
        walls=walls,
        masks=masks,
        lights=np.array([[3., 3.]]),
        res=geometry.RES)

    geometry.display(g)

TODO: Image of geometry

As well as the lights, this :class:`rebar.dotdict.dotdict` gives the locations of the lights and the resolution of the 
mask - which here is the resolution that :func:`megastep.geometry.masks` uses by default.