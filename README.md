# megastep

**This repo is still pre-release. It's public because it makes some bits of testing & distribution setup easier. If you come across it before this warning is removed, don't expect things to work**.

**megastep** helps you build 1-million FPS reinforcement learning environments with a single consumer GPU.

## Examples
Quoted FPS are for a single RTX 2080 Ti and random actions; visualizations are with a 512-neuron LSTM:

**[Deathmatch](megastep/demo/envs/deathmatch.py)**: 150 lines of Python, 1.2m FPS.

**[Explorer](megastep/demo/envs/explorer.py)**: 100 lines of Python, 180k FPS.

**[PointGoal](megastep/demo/envs/waypoint.py)**: 100 lines of Python, 1.1m FPS.

## Features
* Run thousands of environments in parallel, entirely on the GPU.
* Write your own environments using PyTorch alone, no CUDA necessary.
* 1D observations. The world is much more interesting horizontally than vertically.
* One or many agents, and one or many cameras per agent.
* A database of 5000 home layouts to explore, based on [Cubicasa5k](https://github.com/CubiCasa/CubiCasa5k)
* A minimal, modular library. Not a framework.
* Extensive documentation, tutorials and explanations. 

## Setup
**This is a GPU-only package**. 
```
pip install megastep
```
or, for the full demo,
```
pip install megastep[cubicasa,rebar]
```
It's tested on [the 'build' stage of this Docker image](docker/Dockerfile).

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

## Possible Bugs
* Agents seem prone to flying into corners. This might be an accidental behaviour, or it might indicate a problem with the engine. At least one prior version of the Explorer env had an issue with agents learning to fly really hard at walls so they could clip through it and collect the reward for seeing the other side.
* The Deathmatch env learns excruciatingly slowly. This might be due to the sparse reward compared to the Explorer env, or it might be due to a bug in my training algorithm, or it might be that I'm not using population-based training, or it might be bad hyperparameters, or it might be due to a bug in the environment. Reinforcement learning, what fun!

## Alternatives
* **[Sample Factory](https://github.com/alex-petrenko/sample-factory)**
* **[Multiagent Particle Env](https://github.com/openai/multiagent-particle-envs)**

## Commands
* Build docs: `sphinx-build -b html docs docs/_build`
* Serve docs: `docker exec -it megastep python -m http.server --directory /code/docs/_build 9095`
* Build build container: `ssh ajones@aj-server.local "docker build --target build -t onedee:build /home/ajones/code/onedee/docker"`
* Run build container: `docker run -it --rm --name onedeebuild -v /home/ajones/code/onedee:/code onedee:build /bin/bash`