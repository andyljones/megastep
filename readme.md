# megastep

**megastep** helps you build 1-million FPS reinforcement learning environments with a single consumer GPU.

## Examples
Quoted FPS are for a single RTX 2080 Ti and random actions:

**[Deathmatch](onedee/envs/deathmatch.py)**: 150 lines of Python, 250k FPS.

**[Explorer](onedee/envs/explorer.py)**: 100 lines of Python, 1m FPS.

**[PointGoal](onedee/envs/waypoint.py)**: 100 lines of Python, 2m FPS.

## Features
* Run thousands of environments in parallel, entirely on the GPU.
* Write your own environments using PyTorch alone, no CUDA necessary.
* 1D observations. The world is much more interesting horizontally than vertically.
* One or many agents, and one or many cameras per agent.
* A database of 5000 home layouts to explore
* A minimal, modular library. Not a framework.
* Extensive documentation, tutorials and explanations. 

## Documentation
* Tutorials
    * Setting Things Up
    * Writing the Waypoint Environment
    * Writing the Explorer Environment
    * Writing the Deathmatch Environment
    * Training an Agent
    * Writing Your Own CUDA Kernels
* Explanations
    * An Overview Of Megastep
    * Environment Design Best Practices
* API Docs

## Background
Most reinforcement learning setups involve some small number of GPUs to do the learning, and a much larger number of CPUs to the do experience collection. As a researcher, your options are to either rent your hardware in the cloud and [pay through the nose for NVIDIA's cloud GPUs](https://www.digitaltrends.com/computing/nvidia-bans-consumer-gpus-in-data-centers/), or spend a lot of cash building server boxes with all the CPUs you need for experience collection.

The obvious solution is to get rid of either the GPUs or the CPUs. Getting rid of the GPUs isn't really feasible since neural nets are deathly slow without them. Getting rid of the CPUs means writing environments in CUDA, which isn't for the faint of heart. 

Thing is, most RL environments burn their resources - both code and flops - on information that's irrelevant to the experiments you want to conduct. [AirSim](https://microsoft.github.io/AirSim/) is an amazing piece of work, but if you train a nav policy in there you're basically discovering just how inefficiently your agents can learn to see. It turns out that while writing a full sim in CUDA is a bit scary, it doesn't take much work to get something that'll be produce interesting behaviour.

Now follow this logic through to its natural conclusion and you'll find yourself building finite state machines and gridworlds. These though are a bit _too_ simplified to support certain skills - like odometry or geometric reasoning - that we might be interested in.

**megastep** is intended to be midpoint between these two extremes, between full 3D simulators and gridworlds. Thanks to gravity making the world suspiciously flat, many of the behaviours we'd like to investigate in 3D are just as plausible in 2D. And in 2D, things are simple enough that one fool can bolt together a CUDA game engine without breaking a sweat.

## Future Directions
There are many directions that I could plausibly take this project in, but the combination of [The Bitter Lesson](http://incompleteideas.net/IncIdeas/BitterLesson.html), [Scaling Laws for Natural Language Models](https://arxiv.org/pdf/2001.08361.pdf) and [GPT-3](https://arxiv.org/abs/2005.14165) have convinced me that I should aim my efforts at the compute side of things rather than the simulation side of things. 

That's me though! If you're interested in taking megastep forward, some neat things to do:

  * Add better physics. Right now the physics is that there are dynamic circles and static lines, and if two objects collide they stop moving. With better physics, you could plausibly recreate [OpenAI's Hide & Seek](https://openai.com/blog/emergent-tool-use/) work.
  * Demonstrate transfer learning across sims. Can behaviour learned in a fast, cheap simulation like this one be transferred to an expensive sim like [AirSim](https://microsoft.github.io/AirSim/)?
  * Generative geometric modelling. Deepmind have a cool paper on learning priors about the world [from egomotion alone](https://deepmind.com/blog/article/neural-scene-representation-and-rendering). Again, can this be demonstrated on far cheaper hardware if you work in a faster simulator?
  * megastep focuses on geometric simulations - but there's no reason that finite state machine and gridworld envs shouldn't be GPU accelerated too.

I consider megastep to be feature complete, but I'm happy to provide pointers and my own thoughts on these topics to anyone who's interested.

## Possible Bugs
* Agents seem prone to flying into corners. This might be an accidental behaviour, or it might indicate a problem with the engine. At least one prior version of the Explorer env had an issue with agents learning to fly really hard at walls so they could clip through it and collect the reward for seeing the other side.
* The Deathmatch env learns excruciatingly slowly. This might be due to the sparse reward compared to the Explorer env, or it might be due to a bug in my training algorithm, or it might be that I'm not using population-based training, or it might be bad hyperparameters, or it might be due to a bug in the environment. Reinforcement learning, what fun!

## Alternatives
* **[Sample Factory](https://github.com/alex-petrenko/sample-factory)**
* **[Multiagent Particle Env](https://github.com/openai/multiagent-particle-envs)**

## Commands
Build docs: `sphinx-build -b html docs docs/_build`
Serve docs: `docker exec -it onedee python -m http.server --directory /code/docs/_build 9095`