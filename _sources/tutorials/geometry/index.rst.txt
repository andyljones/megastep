.. _custom-geometry:

Custom Geometry
===============
In this tutorial, we're going to create the oh-so-simple :func:`~megastep.toys.box` geometry from scratch. It's not
particularly interesting on it's own, but it does demonstrate a lot of useful tools.

We start by listing the corners of a box in order::

    import numpy as np
    corners = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]])

These corners give a 1m box, which is a bit too small for our purposes. We can scale it up by multiplying by the
width we want. It's also a good idea to shift it 1m up and to the right, as lots of machinery in megastep assumes
that everything happens in the top-right quadrant (ie, above and to the right of the origin). There's no fundamental
reason for this, it just simplifes some stuff internally. ::

    corners = 5*corners + 1

Then to get the walls, we take all sequential pairs of corners and stack them::

    from megastep import geometry
    walls = np.stack(geometry.cyclic_pairs(corners))

You can check that these walls are what we think they are by putting them in a :ref:`dotdict <dotdicts>` and using
:func:`~megastep.geometry.display`::

    geometry.display(dotdict.dotdict(
        walls=walls))

.. image:: walls.png
    :alt: A visualization of the walls
    :width: 640

With the walls in place, the other thing to deal with is rooms. There's no strict definition of a room; they're 
just small, generic regions. The usual use of them is to make it easy to spawn multiple agents near eachother.

In this case, our room is going to just be the corners we had before. That's a list of corners though, while our 
geometry wants a mask. Fortunately there's already a function to turn one into the other::

    masks = geometry.masks(walls, [corners])

Again, we can plot it to check how it looks::

    g = dotdict.dotdict(
        walls=walls,
        masks=masks,
        res=geometry.RES)
    geometry.display(g)

As well as adding the masks themselves to the dict, we've also added the resolution of the mask. In this case, it's
the resolution :func:`~megastep.geometry.masks` uses by default. The result is this:

.. image:: walls-masks.png
    :alt: A visualization of the walls and masks together
    :width: 640

This ``masks`` array has a -1 where there's a wall, a 0 where there's free space, and a 1 where our room is. 

Now that we've got both walls and masks, we just need to add the location of lights::

    from rebar import dotdict
    g = dotdict.dotdict(
        walls=walls,
        masks=masks,
        lights=np.array([[3.5, 3.5]]),
        res=geometry.RES)
    geometry.display(g)

.. image:: complete.png
    :alt: A visualization of the walls and masks together
    :width: 640

And we're done! This is all that's needed to create :ref:`scenery <scenery>` for your environments.

It's mentioned in the :ref:`geometry <geometry>` section but worth re-mentioning here: geometries are dicts rather 
than classes because as you develop your own environments, scene and geometries you'll likely find you have
different ideas about what information a geometry needs to carry around. A dotdict is much easier to modify in that
case than a class.