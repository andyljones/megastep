import matplotlib as mpl
import numpy as np
import torch
from . import common
from rebar.arrdict import tensorify
from rebar import dotdict

def lengths(lines):
    return ((lines[..., 0, :] - lines[..., 1, :])**2).sum(-1)**.5

def agent_frame():
    corners = [
            [-.5, -1.], [+.5, -1.], 
            [+1., -.5], [+1., +.5],
            [+.5, +1.], [-.5, +1.],
            [-1., +.5], [-1., -.5]]
    n = len(corners)
    walls = [[corners[i], corners[(i+1) % n]] for i in range(n)]
    return common.AGENT_WIDTH/2*np.array(walls)

def agent_colors():
    k, g, r = '.25', 'g', 'r'
    colors = (k, g, k, r, k, r, k, g)
    return np.stack([mpl.colors.to_rgb(s) for s in colors])

def resolutions(lines):
    return np.ceil(lengths(lines)/common.TEXTURE_RES).astype(int)

def wall_pattern(n, l=.5, random=np.random):
    p = common.TEXTURE_RES/l
    jumps = random.choice(np.array([0., 1.]), p=np.array([1-p, p]), size=n)
    jumps = jumps*random.normal(size=n)
    value = .5 + .5*(jumps.cumsum() % 1)
    return value

def init_colors(lines, colors, is_agent, random=np.random):
    texwidths = resolutions(lines)
    starts = texwidths.cumsum() - texwidths

    indices = np.full(texwidths.sum(), 0)
    indices[starts] = 1
    indices = np.cumsum(indices) - 1
    textures = common.gamma_decode(colors[indices])

    # Gives walls an even pattern that makes depth perception easy
    pattern = wall_pattern(textures.shape[0], random=random)
    pattern[is_agent[indices]] = 1.
    textures = textures*pattern[:, None]

    return textures, texwidths

@torch.no_grad()
def init_scene_old(cuda, designs, device='cuda', random=np.random): 
    n_agents = designs[0].n_agents

    agentlines = np.tile(agent_frame(), (n_agents, 1, 1))
    agentcolors = np.tile(agent_colors(), (n_agents, 1))

    data = dotdict(frame=agent_frame())
    lights, lightwidths, lines, linewidths, colors, is_agent  = [], [], [], [], [], []
    for d in designs:
        lights.extend([d.lights])
        lightwidths.append(len(d.lights))

        lines.extend([agentlines, d.walls])
        linewidths.append(len(agentlines) + len(d.walls))
        is_agent.extend([np.full(len(agentlines), True), np.full(len(d.walls), False)])

        colors.extend([agentcolors, d.colors])
    data['lights'], data['lightwidths'] = np.concatenate(lights), np.array(lightwidths)
    data['lines'], data['linewidths'] = np.concatenate(lines), np.array(linewidths)
    colors = np.concatenate(colors) 
    is_agent = np.concatenate(is_agent)

    data['textures'], data['texwidths'] = init_colors(data['lines'], colors, is_agent, random)

    scene = cuda.Scene(**tensorify(data).to(device))
    cuda.bake(scene, n_agents)

    return scene

def random_lights(centroids, random=np.random):
    pass

@torch.no_grad()
def init_scene(cuda, plans, n_agents, device='cuda', random=np.random): 
    agentlines = np.tile(agent_frame(), (n_agents, 1, 1))
    agentcolors = np.tile(agent_colors(), (n_agents, 1))

    data = []
    for p in plans:
        lights = random_lights(p.centroids)
        data.append({
            'lights': lights,
            'lightwidths': len(lights),
            'lines': lines,
            'linewidths': len(lines),
            'is_agent': is_agent,
            'colors': colors })
    

