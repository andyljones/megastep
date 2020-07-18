########
Concepts
########

There are some ideas that are referenced in many places in this documentation. 

.. _dotdicts:

dotdicts and arrdicts
=====================
dotdicts and arrdicts are used in many places in megastep in preference to custom classes. 
 * A :class:`~rebar.dotdict.dotdict` is a dictionary with dot (attribute) access to its elements and a bunch of useful behaviours.
 * A :class:`~rebar.arrdict.arrdict` does everything a dotdict does, but with extra support for array/tensor elements.

There are several serious issues with giving up on static typing, but in a research workflow I believe the benefits
outweigh those costs.

Everything below applies to both dotdicts and arrdicts, except for the indexing and binary operation support. Those are 
implemented on arrdicts alone, as they don't make much sense for general elements.

Here're some example dotdicts that'll get exercised in the examples below

>>> objs = dotdict(
>>>     a=dotdict(
>>>         b=1, 
>>>         c=2), 
>>>     d=3)
>>> 
>>> tensors = arrdict(
>>>     a=torch.tensor([1]), 
>>>     b=arrdict(
>>>         c=torch.tensor([2])))

Attribute access
----------------
You can access elements with either ``d[k]`` or ``d.k`` notation, but you must assign new values with ``d[k] = v`` . 

TODO: Make setting with attributes illegal

This convention is entirely taste, but it aligns well with the usual use-case of assigning values rarely and 
reading them regularly.

Forwarded Attributes
--------------------
If you try to access an attribute that isn't a method of ``dict`` or a key in the dotdict, then the attribute access
instead be forwarded to every value in the dictionary. This means if you've got a dotdict full of CPU tensors, you
can send them all to the GPU with a single call:

>>> gpu_tensors = cpu_tensors.cuda()

What's happening here is that the ``.cuda`` access returns a dotdict full of ``.cuda`` attributes. Then the call
itself is forwarded to each tensor, and the results are collected and returned in a tree with the same keys.

Fair warning: be careful not to use keys in your dotdict that clash with the names of methods you're likely to
use.

Method Chaining
---------------
There are a couple of methods on the dotdict itself for making `method-chaining
<https://tomaugspurger.github.io/method-chaining.html>`_ easier. Method chaining is nice because the computation
flows left-to-right, top-to-bottom. For example, with :func:`pipe` you can act on the entire datastructure

>>> objs.pipe(list)
['a', 'd']

or with :func:`map` you can act on the leaves

>>> objs.map(float)
dotdict:
a    dotdict:
    b    1.0
    c    2.0
d    3.0

or with :func:`starmap` you can combine it with another dotdict

>>> objs.starmap(int.__add__, objs)  
dotdict:
a    dotdict:
    b    2
    c    4
d    6

or you can do all these things in sequence

>>> (objs
>>>     .map(float)
>>>     .starmap(float.__add__, d)
>>>     .pipe(list))
['a', 'd']

Pretty-printing
---------------
As you've likely noticed, when you nest dotdicts inside themselves then they're printed prettily:

>>> objs
dotdict:
a    dotdict:
    b    1
    c    2
d    3

It's especially pretty when some of your elements are collections, possibly with shapes and dtypes:

>>> tensors
arrdict:
a    Tensor((1,), torch.int64)
b    arrdict:
     c    Tensor((1,), torch.int64)

Indexing
--------
Indexing is exclusive to arrdicts. On arrdicts, indexing operations are forwarded to the values:

>>> tensors[0]
arrdict:
a    Tensor((), torch.int64)
b    arrdict:
     c    Tensor((), torch.int64)
>>> tensors[0].item()  # the .item() call is needed to get it to print nicely
arrdict:
a    1
b    arrdict:
     c    2

All the kinds of indexing that the underlying arrays/tensors support is supported by arrdict.

Binary operations
-----------------

Binary operation support is also exclusive to arrdicts. You can combine two arrdicts in all the ways you'd combine
the underlying items

>>> tensors + tensors
arrdict:
a    Tensor((1,), torch.int64)
b    arrdict:
     c    Tensor((1,), torch.int64)
>>> (tensors + tensors)[0].item() # the [0].item() call is needed to get it to print nicely
arrdict:
a    2
b    arrdict:
     c    4

It works equally well with Python scalars, arrays, and tensors, and pretty much every binary op you're likely to use
is covered. Call ``dir(arrdict)`` to get a list of the supported magics.

Use cases
---------
You generally use dotdict in places that *really* you should use a :class:`namedtuple`, except that forcing explicit types on
things would make it harder to change things as you go. Using a dictionary instead lets you keep things flexible. The
principal costs are that you lose type-safety, and your keys might clash with method names.

.. _raggeds:

Raggeds
=======
Ragged arrays and tensors are basically arrays-of-arrays, with the values stored in a contiguous backing array to speed up
operations. **megastep** has both numpy and torch Raggeds, and both are created using :func:`~megastep.ragged.Ragged`.

As an example, here's a simple ragged array:: 

    from megastep.ragged import Ragged

    # Subarrays are [0., 1., 2.], [3.], [4., 5.]
    vals = np.array([0., 1., 2., 3., 4., 5.]) 
    widths = np.array([3, 1, 2])
    r = Ragged(vals, widths)

The ``widths`` array gives the widths of each subarray. 

Indexing
--------
Indexing with an integer retrieves the corresponding subarray:

>>> r[0]
array([0, 1, 2])
>>> r[1]
array([3])
>>> r[2]
array([4, 5])

and it can also be sliced:

>>> r[:2]
RaggedNumpy([3 1])

Conversion
----------
Numpy raggeds can be turned back-and-forth into Torch raggeds:

>>> r.torchify()
<megastepcuda.Ragged1D at 0x7fba25320d30>
>>> r.torchify().numpyify()

Be warned that the torch side of things only supports backing tensors with at most 3 dimensions. 

Attributes
----------
If you want to do bulk operations on a ragged, you'll usually want to operate on the backing array directly. There
are a couple of attributes to help with that:

>>> r.vals   # the backing array
array([0., 1., 2., 3., 4., 5.])
>>> r.widths # the subarray widths
array([3, 1, 2])
>>> r.starts # indices of the start of each subarray
array([0, 3, 4])
>>> r.ends   # indices of the end of each subarray
array([3, 4, 6])

Inversion
---------
There is also an ``.inverse`` attribute that tells you which subarray every element of the backing array corresponds to:

>>> r.inverse
array([0, 0, 0, 1, 2, 2])

.. _geometry:

Geometries
==========
A *geometry* describes the static environment that the agents move around in. They're usually created by :mod:`~megastep.cubicasa` 
or with the functions in :mod:`~megastep.toys` , and then passed en masse to an environment or :class:`~megastep.core.Core` .

You can visualize geometries with :mod:`~megastep.geometry.display` :

.. image:: _static/geometry.png
    :alt: A matplotlib visualization of a geometry
    :width: 640

Practically speaking, a geometry is a :ref:`dotdict <dotdicts>` with the following attributes:

id
    An integer uniquely identifying this geometry

walls
    An (M, 2, 2)-array of endpoints of the walls of the geometry, given as (x, y) coordinates in units of meters.

    One 'weird' restriction is that all the coordinates should be strictly positive. This is not a fundamental
    restriction, it just makes a bunch of code elsewhere in megastep simpler if the geometry can be assumed to be in
    the top-right quadrant.

lights
    An (N, 2)-array of the locations of the lights in the geometry, again given as (x, y) coordinates

    As with the walls, the lights should all have strictly positive coordinates.

masks
    An (H, W) masking array describing the rooms and free space in the geometry. 
    
    The mask is aligned with its lower-left corner on (0, 0), and each cell is **res** wide and high. You can map
    between the (i, j) indices of the mask and the (x, y) coords of the walls and lights with
    :func:`~megastep.geometries.centers` and :func:`~megastep.geometries.indices`

    The mask is ``-1`` in cells touching a wall, and otherwise ``0`` in free space or positive integer if the cell is
    in a room. Each room gets its own positive integer. 

res
    A float giving the resolution of **masks** in meters.

The geometry is a dotdict rather than a class because when writing your own environments, it's common to want to nail 
extra bits of information onto the side of the default geometry. That *could* be handled by subclassing, but I have a
personal :ref:`aversion to inheritance hierarchies in research code<inheritance>`.

.. _agents:

Agents
======
TODO: Agents concept

.. _scenery:

Scenery
=======
TODO: Scenery concept

.. _rendering:

Rendering
=========
TODO: Rendering concept

.. _physics:

Physics
=======
TODO: Physics concept

.. _plotting:

Plotting
========
TODO: Plotting concept

.. _decision-world:

Decision & World
================
TODO: Decision/world concept

.. _models:

Models
======
TODO: Models concept

.. _subpackages:

Subpackages
===========
TODO: Subpackages concept

.. _spaces:

Spaces & Heads
==============
TODO: Spaces and heads concept

.. _patterns:

Env Patterns
=============
TODO: Env patterns concept

* Use modules where possible
* Get your plotter working first
* A display method
* A _reset method
* A _observe method
* n_envs and device