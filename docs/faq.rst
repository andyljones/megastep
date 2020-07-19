.. _faq:

===
FAQ
===

.. _inheritance:

Why doesn't megastep use inheritance?
-------------------------------------
A general adage in software is to prefer `composition over inheritance <https://stackoverflow.com/questions/49002/prefer-composition-over-inheritance>`_.
It's a good rule of thumb, but I feel that of the realities of research code make the preference even more extreme.

Research code is a very unusual kind of code. It's `written many times and read once (if ever) <https://devblogs.microsoft.com/oldnewthing/20070406-00/?p=27343>`_,
it's typically written by one person in a short period of time and it's typically only a few thousand lines of code
that are understood inside and out. Because of this, researchers can happily trade off a lot of otherwise-good
development practices in favour of iteration velocity - the ability to adapt your codebase to a new idea quickly and
easily.

Since megastep is explicitly intended to be a foundation for research, flexibility and iteration velocity feel far more 
important than the robustness you get from inheritance. 

.. _openai-gym:

Why don't you use the OpenAI Gym interface?
-------------------------------------------
There are a couple of ways in which megastep departs from the `Gym interface <https://gym.openai.com/docs/#environments>`_.

The first way is that all the observations, rewards, and resets are vectorized. This is necessary, as megastep is 
naturally vectorized in a way that the Gym envs aren't. 

The second, more debatable way is that the Gym returns observations, rewards and resets as a tuple, and takes actions. 
megastep meanwhile :ref:`passes dicts of these things in both directions <decision-world>`. The advantage of this is
opacity: if you want to pass some extra information between env and agent - the most common kind being when a reset 
occurs so that the agent can clear its memory - it's just an extra key in the dict. The experience collection loop 
that mediates between env and agent doesn't need to know anything about it. 

Writing a shim that turns any megastep env into an Gym env should be easy enough if you're so inclined.

.. _cubicasa-license:

What's with the cubicasa license?
---------------------------------
The cubicasa dataset - the dataset of 5000 home layouts - is derived from the `Cubicasa5k <https://github.com/CubiCasa/CubiCasa5k>`_ 
dataset. This dataset was released under a CreativeCommons Non-Commercial License, while megastep as a whole is under a 
MIT license. Since the cubicasa dataset in this project is a heavily-modified version of the original dataset, I think
it could be plausibly considered `transformative use <https://www.copyright.gov/fair-use/more-info.html#:~:text=Transformative%20uses%20are%20those%20that,purpose%20of%20encouraging%20creative%20expression.>`_
and so be re-released under an MIT license. But as an independent researcher with no legal team, I can't risk claiming 
that. Rather I've emailed Cubicasa and asked for their blessing on this interpretation.

In the meantime though, downloading the cubicasa dataset is hidden behind a is-this-commercial-use prompt. Not ideal,
but the best I could come up with.

If you would like to use megastep for commercial purposes, you are absolutely welcome to - just use a different geometry
sampler to the default one. There are the :mod:`~megastep.toys` geometries already available, and writing a maze 
generator should be fairly simple - just output a dict conforming :ref:`to the spec <geometry>`.

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
   accelerated too. 
 * 1D observations are small enough to stick your replay buffer on the GPU. With 64-pixel 3-color
   half-precision observations, you can fit 2.5m obs per GB. Can this be used to eke extra performance out of
   off-policy algorithms?

I consider megastep to be feature complete, but I'm happy to provide pointers and my own thoughts on these topics to
anyone who's interested in forking it to build something greater.

What are some alternatives to megastep?
---------------------------------------
 * `Sample Factory <https://github.com/alex-petrenko/sample-factory>`_
 * `Multiagent Particle Env <https://github.com/openai/multiagent-particle-envs>`_
 * `VizDoom <https://github.com/mwydmuch/ViZDoom>`_
 * `dmlab30 <https://github.com/deepmind/lab>`_

What about other OSes?
----------------------
If you're on a different OS, then it's possible megastep will work, but I can't provide you any support. You're welcome
to ask for help on the GitHub issues page, but you'll be relying on the community to come up with an answer.

What if I don't have CUDA?
--------------------------
If you haven't got CUDA, megastep will not work. There are some parts of megastep - like the cubicasa package - 
that you may still find useful, but in that case I recommend just copy-pasting the code you want from Github.

How can I install *just* megastep?
----------------------------------
The default :ref:`install <install>` pulls in everything needed to run the demos and tutorials. If you want something
minimal::

    pip install megastep

ie, omit the bit in square brackets. You can read more about what's missing in the :ref:`subpackages <subpackages>`
section.