===
FAQ
===

.. _install:

How do I install megastep?
--------------------------
If you're working on an Ubuntu machine with CUDA already set up, it should be as simple as ::

    pip install git+https://github.com/andyljones/megastep.git#egg=megastep[cubicasa,rebar]

This installs everything needed to run the demos and tutorials. If you want :ref:` something minimal <subpackages>` ::

    pip install git+https://github.com/andyljones/megastep.git#egg=megastep[cubicasa,rebar]

There are some fairly hefty dependencies so it might take a while to install, but it should get there fine. If it
doesn't, my total guess is that it's the Pytorch install that will be the problem, in which case I refer you to
`their install page <https://pytorch.org/get-started/locally/>`_. 

**The first time you import megastep will be slow** as it compiles the C++ side of things.

**If you haven't got CUDA, megastep will not work**. There are some parts of megastep - like the cubicasa package - 
that you may still find useful, but in that case I recommend just copy-pasting the code you want from Github.

If you're on a different OS, then it's possible megastep will work, but I can't provide you any support. You're welcome
to ask for help on the GitHub issues page, but you'll be relying on the community to come up with an answer.

.. _inheritance:

Why doesn't megastep use inheritance?
-------------------------------------
TODO: Figure out a better way to phrase this

One way to think about inheritance in software development is that's about offering a secondary interface to a class.

The primary interface to a class is its public methods. When you write a class for some other part of your program to 
use, this is usually what you have in mind.

When you encourage people to inherit from your classes though, you're effectively declaring a secondary interface,
saying 'here are some useful ways to exploit its private state'.

The thing is, Python's ideas of public and private amount to gentle suggestions. This leads to a tertiary interface
to every class, which is where you totally ignore what the designer of the class intended you to do and rely instead
entirely on how the class *actually works*. You freely read and write its private state, monkey patch its methods and
generally let slip the dogs of terrible software development.

Researchers are very fond of this tertiary wild-west interface. The reason researchers are fond of it is either
because they're terrible developers (the popular answer), or because research code is a very unusual kind of code.
It's `written many times and read once (if ever) <https://devblogs.microsoft.com/oldnewthing/20070406-00/?p=27343>`_,
it's typically written by one person in a short period of time and it's typically only a few thousand lines of code
that are understood inside and out. Because of this, researchers can happily trade off a lot of otherwise-good
development practices in favour of iteration velocity - the ability to adapt your codebase to a new idea quickly and
easily.

Since **megastep** is explicitly intended to be a foundation for research, it's designed with the third interface in mind.
There are few private methods, and any state that is likely interesting to a user is there for the taking.

.. _openai-gym:

Why don't you use the OpenAI Gym interface?
---------------------------------------------


.. _cubicasa-license:

What's with the cubicasa license?
---------------------------------


.. _why:

Why did you write megastep?
---------------------------
Most reinforcement learning setups involve some small number of GPUs to do the learning, and a much larger number of
CPUs to the do experience collection. As a researcher, your options are to either rent your hardware in the cloud and
`pay through the nose for NVIDIA's cloud GPUs <https://www.digitaltrends.com/computing/nvidia-bans-consumer-gpus-in-data-centers/>`_, 
or spend a lot of cash building server boxes with all the CPUs you need for experience collection.

The obvious solution is to get rid of either the GPUs or the CPUs. Getting rid of the GPUs isn't really feasible
since neural nets are deathly slow without them. Getting rid of the CPUs means writing environments in CUDA, which
isn't for the faint of heart.

Thing is, most RL environments burn their resources - both code and flops - on information that's irrelevant to the
experiments you want to conduct. `AirSim <https://microsoft.github.io/AirSim/>`_ is an amazing piece of work, but if
you train a nav policy in there you're basically discovering just how inefficiently your agents can learn to see. It
turns out that while writing a full sim in CUDA is a bit scary, it doesn't take much work to get something that'll be
produce interesting behaviour.

Now follow this logic through to its natural conclusion and you'll find yourself building finite state machines and
gridworlds. These though are a bit _too_ simplified to support certain skills - like odometry or geometric reasoning
- that we might be interested in.

**megastep** is intended to be midpoint between these two extremes, between full 3D simulators and gridworlds. Thanks
to gravity making the world suspiciously flat, many of the behaviours we'd like to investigate in 3D are just as
plausible in 2D. And in 2D, things are simple enough that one fool can bolt together a CUDA game engine without
breaking a sweat.

Where might this go in future?
------------------------------
There are many directions that I could plausibly take this project in, but the combination of `The Bitter
Lesson <http://incompleteideas.net/IncIdeas/BitterLesson.html>`_, `Scaling Laws for Natural Language
Models <https://arxiv.org/pdf/2001.08361.pdf>`_ and `GPT-3 <https://arxiv.org/abs/2005.14165>`_ have convinced me that I
should aim my efforts at the compute side of things rather than the simulation side of things.

That's me though! If you're interested in taking megastep forward, here are some research directions I had queued up:
 * Add better physics. Right now the physics is that there are dynamic circles and static lines, and if two objects
   collide they stop moving. With better physics, you could plausibly recreate `OpenAI's Hide & Seek <https://openai.com/blog/emergent-tool-use/>`_
   work. Demonstrate transfer learning across sims. Can behaviour learned in a fast, cheap simulation like this one
   be transferred to an expensive sim like `AirSim <https://microsoft.github.io/AirSim/>`_?
 * Generative geometric modelling. Deepmind have a cool paper on learning priors about the world `from egomotion alone <https://deepmind.com/blog/article/neural-scene-representation-and-rendering>`_. 
   Again, can this be demonstrated on far cheaper hardware if you work in a faster simulator? 
 * megastep focuses on geometric simulations - but there's no reason that finite state machine and gridworld envs shouldn't be GPU
   accelerated too. 1D observations are small enough to stick your replay buffer on the GPU. With 64-pixel 3-color
   half-precision observations, you can fit 2.5m obs per GB. Can this be used to eke extra performance out of
   off-policy algorithms?

I consider megastep to be feature complete, but I'm happy to provide pointers and my own thoughts on these topics to
anyone who's interested in forking it to build something greater.

What are some alternatives to megastep?
---------------------------------------
 * `Sample Factory <https://github.com/alex-petrenko/sample-factory>`_
 * `Multiagent Particle Env <https://github.com/openai/multiagent-particle-envs>`_
