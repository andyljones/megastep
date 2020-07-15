====================
Writing a Simple Env
====================

In this tutorial we're going to write the simplest possible environment. It's going to have one agent that gets
rewarded for colliding with a wall. Not a particularly interesting task, but it'll be easy to follow, and give us a
base of techniques that we can later build more interesting environments on top of.

TODO: Animation of collision env

This tutorial leads with a view-from-a-thousand-foot doing-without-understanding version. There are links throughout
that explain what's going on in detail. 

The High Level
--------------
The place to start is with the :ref:`geometry <geometry>`. The geometry describes the walls, rooms and lights of
an environment. In later tutorials we're going to generate thousands of unique geometries, but here for our
simplest-possible env, a single geometry will do. A single, simple geometry::

    form megastep import geometry, toys

    g = toys.box()
    geometry.display(g)

TODO: Image of box env

Yup, it's a box. Four walls and one room. There's :ref:`more below about how the geometry is made <simple-env-geometry>`,
and also a :ref:`brief discussion of its place in megastep <geometry>`.

A geometry on its own is not enough for the renderer to go on though. For one it's missing texture, and for two it only 
describes a single environment, when megastep's key advantage is the simulation of thousands of environments in parallel.
To turn the geometry into something the renderer can use, we turn it into a :class:`megastep.cuda.Scene`::

    from megastep import scenery, plotting
    scene = scenery.scene(1024*[g])

    plotting.display(scene, e=126)

TODO: Image of scene

This code generates



.. _simple-env-geometry:

Geometry
--------
To create the box geometry, we start with the corners in order::

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

You can check that these walls are what we think they are by putting them in a :ref:`dotdict <dotdicts>` and using
:func:`megastep.geometry.display`::

    geometry.display(dotdict.dotdict(
        walls=walls))

TODO: Image of walls

With the walls in place, the other thing to deal with is rooms. There's no strict definition of a room; they're 
just small, generic regions. The usual use of them is to make it easy to spawn multiple agents near eachother.

In this case, our room is going to just be the corners we had before. That's a list of corners though, while our 
geometry wants a mask. Fortunately there's already a function to turn one into the other::

    masks = geometry.masks(walls, [corners])

Again, we can plot it to check how it looks::

    geometry.display(dotdict.dotdict(
        walls=walls,
        masks=masks))

TODO: Image of walls and masks

This ``masks`` array has a -1 where there's a wall, a 0 where there's free space, and a 1 where our room is. Now that
we've got both walls and masks, we just need to add the location of lights and some metadata:

    from rebar import dotdict
    g = dotdict.dotdict(
        walls=walls,
        masks=masks,
        lights=np.array([[3., 3.]]),
        id="box",
        res=geometry.RES)
    geometry.display(g)

TODO: Image of geometry

The metadata is an ID - which isn't particularly useful for our single geometry, but is a lot more useful when
generating thousands of them - and the resolution of the mask, which here is the resolution that
:func:`megastep.geometry.masks` uses by default.

It's mentioned in the :ref:`geometry <geometry>` section but worth re-mentioning here: geometries are dicts rather 
than classes because as you develop your own environments, scenery and geometries you'll likely find you have
different ideas about what information a geometry needs to carry around. A dotdict is much easier to modify in that
case than a class.