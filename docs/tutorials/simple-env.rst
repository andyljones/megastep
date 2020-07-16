============================
Writing a Simple Environment
============================

In this tutorial we're going to write the simplest possible environment. It's going to have one agent that gets
rewarded for colliding with a wall. Not a particularly interesting task, but it'll be easy to follow, and give us a
base of techniques that we can later build more interesting environments on top of.

TODO: Animation of collision env

The High Level
--------------
This tutorial leads with a view-from-a-thousand-foot doing-without-understanding version, but there are links throughout
that explain what's going on in detail. 

It'd a good idea to open yourself a `Jupyter notebook <https://jupyter.org/install>`_ or `colab
<https://colab.research.google.com/>`_ and copy-paste the code as you go, experimenting with anything you don't quite
understand.

Once you've got your notebook up and running, the first thing to do is to :ref:`install megastep <install>`,
and the extra demo dependencies, which should be as simple as running ::

    !pip install megastep[rebar] --yes

in a cell.

Geometry
********
The place to start is with the :ref:`geometry <geometry>`. The geometry describes the walls, rooms and lights of
an environment. In later tutorials we're going to generate thousands of unique geometries, but here for our
simplest-possible env, a single geometry will do. A single, simple geometry::

    from megastep import geometry, toys

    g = toys.box()
    geometry.display(g)

TODO: Image of box env

Yup, it's a box. Four walls and one room. There's :ref:`more below about how the geometry is made <simple-env-geometry>`,
and also a :ref:`brief discussion of its place in megastep <geometry>`.

Scenery
*******
A geometry on its own is not enough for the renderer to go on though. For one it's missing texture, and for two it only 
describes a single environment, when megastep's key advantage is the simulation of thousands of environments in parallel.
To turn the geometry into something the renderer can use, we turn it into a :class:`megastep.cuda.Scenery`::

    from megastep import scene
    geometries = 128*[toys.box()]
    scenery = scene.scenery(geometries, n_agents=1)

    scene.display(scenery, e=126)

TODO: Image of scenery

This code creates scenery for 1024 copies of our box geometry, each with a randomly-chosen colourscheme and texture.
One copy is shown. You'll notice an :ref:`agent has also been created and placed at the origin <models>`. If you want
to know more about what's going on here, there's :ref:`another brief discussion about scenery <scenery>` and :ref:`a
tutorial on writing your own scenery generator <tutorial-scenery>`.

Rendering
*********
With the scenery in hand, the next thing to do is create a :class:`megastep.core.Core`::

    from megastep import core
    c = core.Core(scenery)

The Core doesn't actually do very much; there're little code in it and all its variables are public. It does do some
setup for you, but after that it's just a bag of useful attributes that you're going to pass to the physics and rendering
engines. 

One of things the core sets up is the :class:`megastep.cuda.Agents` datastructure, which stores where the agents are.
You can take a look with

>>> import torch
>>> c.agents.positions
tensor([[[0., 0.]],
        ... 
        [[0., 0.]]], device='cuda:0')

but all it's going to tell you is that they're at the origin. megastep stores all its state in PyTorch tensors like 
these, and it's a-okay to update them on the fly. By default the origin is outside the box we've built, so as a 
first step let's put them inside the box ::

    c.agents.positions[:] = torch.as_tensor([3., 3.], device=c.device)

And now we can render the agents' view :: 

    from megastep import cuda
    r = cuda.render(c.scenery, c.agents)

This ``r`` is a :class:`megastep.cuda.Render` object, which contains a lot of useful information that you can exploit
when desiging environments. Principally, it contains what the agents see ::

    im = (r.screen
            [[0]]            # get the screen for agents in env #0
            .cpu().numpy())  # move them to cpu & numpy
    plotting.plot_images({'rgb': im}, transpose=True, aspect=.1)

TODO: Plotted image

This is a 1-pixel-high image out from the front of the agent. You can read more about the rendering system in :ref:`this
section <rendering>`. As well as filling up the Render object, calling render does something else: it updates the
agents' models to match their positions. Having moved all the agents to (3, 3) earlier by assigning to
``c.agents.positions``, plotting the scenery again shows that the agents' models have moved from the origin to (3, 3):

    scenery.display(scene)

TODO: Moved image

Physics
*******
Along with :func:`megastep.cuda.render`, the other important call in megastep is :func:`megastep.cuda.physics`. This
call handles moving agents based on their velocities, and deals with any collisions that happen. If we set the agents'
velocities to some obscene value, then make the physics call:

>>> c.agents.momenta[:] = torch.as_tensor([1000., 0.], device=c.device)
>>> p = cuda.physics(c.scenery, c.agents)
>>> c.agents.positions
tensor([[[5.8649, 3.0000]],
        ...
        [[5.8649, 3.0000]]], device='cuda:0')

we see that afterwards, the agents positions have been updated to *roughly* where the right wall is. If we check the 
scenery right now though, the agents' models will still be at (3, 3) however. To update them, we need to call render
again:: 

    cuda.render(c.scenery, c.agents)
    scene.display(c.scenery)

TODO: Updated position

A Skeleton
**********
We've now illustrated the basic loop in megastep::

    g = toys.box()
    scenery = scene.scenery(n_envs*[g], n_agents=1)
    c = cuda.Core(scenery)

    # set agent location
    r = cuda.render(c.scenery, c.agents)
    # generate an observation and send it to the agent
    while True:
        # process decisions from the agent
        p = cuda.physics(c.scenery, c.agents)
        # post-collision alterations
        r = cuda.render(c.scenery, c.agents)
        # generate an observation and send it to the agent

This loop will be hiding at the bottom of any environment you write. For the purposes of actually *using* the environment
though, that 'while' loop needs to be abstracted away. The typical way to do this follows from the `OpenAI Gym
<http://gym.openai.com/docs/#environments>`_, and while we're :ref:`not going to follow their interface exactly
<openai-gym>` we are going to steal the ideas of a 'reset' method and a 'step' method::

    class Collisioneer:

        def __init__(self):
            geometries = 128*[toys.box()]
            scenery = scene.scenery(geometries, n_agents=1)
            self.c = cuda.Core(scenery)

        def reset(self):
            # set agent location
            r = cuda.render(self.c.scenery, self.c.agents)
            # generate an observation and send it to the agent
            return world

        def step(self, decision):
            # process decisions from the agent
            p = cuda.physics(self.c.scenery, self.c.agents)
            # post-collision alterations
            r = cuda.render(self.c.scenery, self.c.agents)
            # generate an observation and send it to the agent
            return world

This is exactly the same code as was in the loop, just with the interation with the agent made explicit through
:ref:`'decision' and 'world' variables <decision-world>`. This is very my syntactic sugar for agent-env interactions,
and while I think it works well, you're free to replace with your own. With this sugar though, the loop becomes much
more flexible::

    env = Collisioneer()
    world = env.reset()
    while True:
        decision = agent(world)
        world = env.step(decision)

Now all that's left to be done is to fill out those comment lines.

An Aside
********
Now that we're building up a class, it's going to be impractical for me to copy-paste the source every time I discuss
a change. Instead, you should grab the completed Collisioneer class from megastep's demo repo::

    from megastep.demo.envs.collisioneer import *
    self = Collisioneer()
    world = self.reset()

The remainder of the code segments will be small 'experiments' - for want of a better word - you can run on this env
to understand what's happening and why it's set up the way it is. If you want to play with the class's definition, 
then open an editor at ``self.__file__`` and copy-paste the contents into your notebook.

(You could alternatively edit it in-place, or copy it into a file ofyour own. Both of those however either require
restarting the kernel after each edit, or setting `autoreload
<https://ipython.org/ipython-doc/3/config/extensions/autoreload.html>`_ up. Autoreload is magical and absolutely
worth your time, but it is a tangent from this tutorial)

Spawning
********
Back to those comment lines! It's a good idea to work through them in order, since that means you can validate that
things are working as expected as you go. The first comment line is to 'set agent location'. We're going to want to
do this on the first reset, and then every time the agent collides with something and needs to be respawned at a new
location.

This is a pretty common task when building an environment, and so there's a :class:`megastep.modules.RandomSpawns`
module to do it for you. It gets added to the env in ``__init__``, ::

    from megastep import modules
    self.spawner = modules.RandomSpawns(geometries, self.core)

and then you can call it with a mask of the agents you'd like to be respawned::

    reset = self.core.agent_full(True)
    self.spawner(reset)

As an aside, the :method:`megastep.core.Core.agent_full` and :method:`megastep.core.Core.env_full` methods will create
on-device tensors for you of shape (n_env, n

This will move each agent to a random position in the room. You can see this directly by inspecting ``self.core.agents.positions``,
or you can render and display it::

    self.core.render(self.core.scenery, self.core.agents)
    scenery.display(self.core.scenery)

TODO: Respawned agent

You can read more about how the respawning module works in the :class:`megastep.modules.RandomSpawns` documentation.

Observations
************
The next comment is 'generate an observation and send it to the agent'. For our find-a-wall-and-collide-with-it, the 
only observation the agent will need is depth, and again there's a module for that::

    self.depth = modules.Depth(self.core)

This time, calling it gives you back a (n_env, n_agent, 1, res, 1)-tensor, suitable for passing to a PyTorch convnet ::

    obs = self.depth()

The render method is called internally by ``depth``, saving us from having to do it explicitly ourselves. 

Following the :ref:`decision-and-world <decision-world>` setup, this obs gets wrapped in a
:class:`rebar.arrdict.arrdict` so that if we decide to nail any other information onto the side of our observations,
it's easy to do so. That means our ``reset`` method in all its glory is ::

    def reset(self):
        self.spawner(self.core.agent_full(True))
        return arrdict.arrdict(obs=self.depth())

>>> self.reset()
arrdict:
obs    Tensor((128, 1, 1, 1, 64), torch.float32)

Actions
*******
The third comment is 'process decisions from the agent'. In our environment the action is simply whether to move 
forward/backward, left/right, or turn left/right. Once again, there's a module for this::

    self.movement = modules.SimpleMovement(self.core)

In the :ref:`decision-and-world <decision-world>` setup, the agent produces a ``decision`` arrdict with an
``"actions"`` key. The :class:`megastep.modules.SimpleMovement` module expects the actions to be an integer tensor,
with values between 0 and 7. Each integer corresponds to a different movement. We can mock a decisions dict easily
enough::

    decision = arrdict.arrdict(actions=self.core.agent_full(3))

and calling the movement module will shift the agents forward::

    self.movement(decision)

As with the ``depth`` module, the ``movement`` module makes the ``physics`` call internally, again saving us from having
to do it ourselves.


Animation
*********
While the tempting thing to do is to start filling out those comment lines, writing environments can be tricky and 
it's worth first doing some work to make bugs more obvious. One of the most powerful ways to spot bugs is to watch
an agent interact with the environment.


.. _simple-env-geometry:

Geometry - in detail
--------------------
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
we've got both walls and masks, we just need to add the location of lights and the resolution of the mask::

    from rebar import dotdict
    g = dotdict.dotdict(
        walls=walls,
        masks=masks,
        lights=np.array([[3., 3.]]),
        res=geometry.RES)
    geometry.display(g)

TODO: Image of geometry

Here, the resolution is the one that :func:`megastep.geometry.masks` uses by default.

It's mentioned in the :ref:`geometry <geometry>` section but worth re-mentioning here: geometries are dicts rather 
than classes because as you develop your own environments, scene and geometries you'll likely find you have
different ideas about what information a geometry needs to carry around. A dotdict is much easier to modify in that
case than a class.