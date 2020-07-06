# megastep

**megastep** helps you build 1-million FPS reinforcement learning environments.

## Examples
Quoted FPS are for a single RTX 2080 Ti and a 512-neuron LSTM agent:

**[Deathmatch](onedee/envs/deathmatch.py)**: 150 lines of Python, 250k FPS with random actions, 40k FPS with an agent:

**[Exploration](onedee/envs/exploration.py)**: 100 lines of Python, 1m FPS with random actions, 40k FPS with an agent:

**[PointGoal](onedee/envs/waypoint.py)**: 100 lines of Python, 2m FPS with random actions, 40k FPS with an agent:

## Features
* A minimal, modular library. Not a framework.
* Run thousands of environments in parallel, entirely on the GPU.
* Work entirely in PyTorch. Only the renderer and physics engine are in C++.
* 1D observations. The world is much more interesting horizontally than vertically.
* One or many agents, and one or many cameras per agent.
* Extensive documentation, tutorials and how-tos. 

## Commands
Build docs: `sphinx-build -b html docs docs/_build`
Serve docs: `docker exec -it onedee python -m http.server --directory /code/docs/_build 9095`