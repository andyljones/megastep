"""TODO-DOCS Scene docs"""
import matplotlib as mpl
import numpy as np
import torch
from . import core, ragged, plotting
from rebar import arrdict
import matplotlib.pyplot as plt

# Ten bland colors from https://medialab.github.io/iwanthue/
COLORS = [
    "#c185ae",
    "#73a171",
    "#5666a4",
    "#9f7c4a",
    "#809cd5",
    "#566e40",
    "#8e537b",
    "#4f9fa4",
    "#b56d66",
    "#5a728c"]

def lengths(lines):
    return ((lines[..., 0, :] - lines[..., 1, :])**2).sum(-1)**.5

def agent_model():
    corners = [
            [-.5, -1.], [+.5, -1.], 
            [+1., -.5], [+1., +.5],
            [+.5, +1.], [-.5, +1.],
            [-1., +.5], [-1., -.5]]
    n = len(corners)
    walls = [[corners[i], corners[(i+1) % n]] for i in range(n)]
    return core.AGENT_WIDTH/2*np.array(walls)

def agent_colors():
    k, g, r = '.25', 'g', 'r'
    colors = (k, g, k, r, k, r, k, g)
    return np.stack([mpl.colors.to_rgb(s) for s in colors])

def resolutions(lines):
    return np.ceil(lengths(lines)/core.TEXTURE_RES).astype(int)

def wall_pattern(n, l=.5, random=np.random):
    p = core.TEXTURE_RES/l
    jumps = random.choice(np.array([0., 1.]), p=np.array([1-p, p]), size=n)
    jumps = jumps*random.normal(size=n)
    value = .5 + .5*(jumps.cumsum() % 1)
    return value

def init_textures(agentlines, agentcolors, walls, random=np.random):
    colormap = np.array([mpl.colors.to_rgb(c) for c in COLORS])
    wallcolors = colormap[np.arange(len(walls)) % len(colormap)]
    colors = np.concatenate([agentcolors, wallcolors])

    texwidths = resolutions(np.concatenate([agentlines, walls]))
    starts = texwidths.cumsum() - texwidths

    indices = np.full(texwidths.sum(), 0)
    indices[starts] = 1
    indices = np.cumsum(indices) - 1
    textures = core.gamma_decode(colors[indices])

    # Gives walls an even pattern that makes depth perception easy
    pattern = wall_pattern(textures.shape[0], random=random)
    pattern[:sum(texwidths[:len(agentlines)])] = 1.
    textures = textures*pattern[:, None]

    return textures, texwidths

def random_lights(lights, random=np.random):
    return np.concatenate([
        lights,
        random.uniform(.5, 2., (len(lights), 1))], -1)

@torch.no_grad()
def scenery(geometries, n_agents=1, device='cuda', random=np.random): 
    agentlines = np.tile(agent_model(), (n_agents, 1, 1))
    agentcolors = np.tile(agent_colors(), (n_agents, 1))

    data = []
    for g in geometries:
        lights = random_lights(g.lights)
        lines = np.concatenate([agentlines, g.walls])
        textures, texwidths = init_textures(agentlines, agentcolors, g.walls, random) 
        data.append(arrdict.arrdict(
            lights=arrdict.arrdict(vals=lights, widths=len(lights)),
            lines=arrdict.arrdict(vals=lines, widths=len(lines)),
            textures=arrdict.arrdict(vals=textures, widths=texwidths)))
    data = arrdict.torchify(arrdict.cat(data)).to(device)
    
    lights = ragged.Ragged(**data['lights'])
    scenery = core.cuda.Scenery(
        n_agents=n_agents,
        lights=lights,
        lines=ragged.Ragged(**data['lines']),
        textures=ragged.Ragged(**data['textures']),
        model=arrdict.torchify(agent_model()).to(device))
    core.cuda.bake(scenery)

    return scenery

def display(scenery, e=0):
    ax = plt.axes()

    state = arrdict.numpyify(
        arrdict.arrdict(
            scenery=scenery.state(e)))

    plotting.plot_lines(ax, state, zoom=False)
    plotting.plot_lights(ax, state)

    plotting.adjust_view(ax, state, zoom=False)

    return ax.figure