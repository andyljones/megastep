########
Concepts
########

There are some ideas that are referenced in many places in this documentation. 

.. _dotdicts:

dotdicts and arrdicts
=====================
dotdicts and arrdicts are used in many places in megastep in preference to custom classes. There are several serious 

.. _geometry:

Geometries
==========
A *geometry* describes the static environment that the agents move around in. They're usually created by :mod:`megastep.cubicasa` 
or with the functions in :mod:`megastep.toys` , and then passed en masse to an environment or :class:`megastep.core.Core` .

You can visualize geometries with :mod:`megastep.geometry.display` :

.. image:: _static/geometry.png
    :alt: A matplotlib visualization of a geometry
    :width: 400

Practically speaking, a geometry is a :ref:`dotdict <dotdicts>` with the following attributes:

id
    An integer uniquely identifying this geometry

walls
    An (M, 2, 2)-array of endpoints of the walls of the geometry, given as (x, y) coordinates in units of meters.

lights
    An (N, 2)-array of the locations of the lights in the geometry, again given as (x, y) coordinates

masks
    An (H, W) masking array describing the rooms and free space in the geometry. 
    
    The mask is aligned with its lower-left corner on (0, 0), and each cell is **res** wide and high. You can map
    between the (i, j) indices of the mask and the (x, y) coords of the walls and lights with
    :func:`rebar.geometries.center_coords` and :func:`rebar.geometries.indices`

    The mask is ``-1`` in cells touching a wall, and otherwise ``0`` in free space or positive integer if the cell is
    in a room. Each room gets its own positive integer. 

res
    A float giving the resolution of **masks** in meters.