# megastep

**megastep** helps you build 1-million FPS reinforcement learning environments with a single consumer GPU.

## Examples
Quoted FPS are for a single RTX 2080 Ti and a 512-neuron LSTM agent:

**[Deathmatch](onedee/envs/deathmatch.py)**: 150 lines of Python, 250k FPS with random actions, 40k FPS with an agent:

**[Explorer](onedee/envs/explorer.py)**: 100 lines of Python, 1m FPS with random actions, 40k FPS with an agent:

**[PointGoal](onedee/envs/waypoint.py)**: 100 lines of Python, 2m FPS with random actions, 40k FPS with an agent:

## Features
* A minimal, modular library. Not a framework.
* Run thousands of environments in parallel, entirely on the GPU.
* Work entirely in PyTorch. Only the renderer and physics engine are in C++.
* 1D observations. The world is much more interesting horizontally than vertically.
* One or many agents, and one or many cameras per agent.
* Extensive documentation, tutorials and explanations. 

## Documentation
* Tutorials
    * Setting Things Up
    * Writing the Waypoint Environment
    * Writing the Explorer Environment
    * Writing the Deathmatch Environment
    * Training an Agent
* How-Tos
    * Environment Design Best Practices
    * Writing Your Own Kernels
* API Docs

## Background
Most reinforcement learning setups involve some small number of GPUs to do the learning, and a much larger number of CPUs to the do experience collection. As a researcher, your options are to either rent your hardware in the cloud and [pay through the nose for NVIDIA's cloud GPUs](https://www.digitaltrends.com/computing/nvidia-bans-consumer-gpus-in-data-centers/), or spend a lot of cash building server boxes with all the CPUs you need for experience collection.

Thing is, most RL environments you'd run on those CPUs burn their resources - both code and flops - on information that's irrelevant to the experiments you want to conduct. [AirSim](https://microsoft.github.io/AirSim/) for example is an amazing piece of work, but if you train a nav policy in there you're mostly discovering just how inefficiently you can learn to see.

That said, chucking out visual observations entirely is a bridge too far. There are interesting skills - like odometry or geometric reasoning - that just can't be learned in a gridworld or with vector-based observations. 

**megastep** is intended to be midpoint between these two extremes, between full 3D simulators and gridworlds. Thanks to gravity making the world suspiciously flat, many of the behaviours we'd like to investigate in 3D are just as plausible in 2D. And in 2D, things are simple enough that one fool can bolt together a game engine without breaking a sweat.

## Possible Bugs
* Agents seem prone to flying into corners. This might be an accidental behaviour, or it might indicate a problem with the engine. At least one prior version of the Explorer env had an issue with agents learning to fly really hard at walls so they could clip through it and collect the reward for seeing the other side.
* The Deathmatch env learns excruciatingly slowly. This might be due to the sparse reward compared to the Explorer env, or it might be due to a bug in my training algorithm, or it might be bad hyperparameters, or it might be due to a bug in the environment. Reinforcement learning, what fun!

## Commands
Build docs: `sphinx-build -b html docs docs/_build`
Serve docs: `docker exec -it onedee python -m http.server --directory /code/docs/_build 9095`