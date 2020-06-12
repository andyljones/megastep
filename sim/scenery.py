import matplotlib as mpl
import numpy as np
import torch
from . import common
from rebar.arrdict import tensorify
from rebar import dotdict

def lengths(lines):
    return ((lines[..., 0, :] - lines[..., 1, :])**2).sum(-1)**.5

def drone_frame():
    corners = [
            [-.5, -1.], [+.5, -1.], 
            [+1., -.5], [+1., +.5],
            [+.5, +1.], [-.5, +1.],
            [-1., +.5], [-1., -.5]]
    n = len(corners)
    walls = [[corners[i], corners[(i+1) % n]] for i in range(n)]
    return common.DRONE_WIDTH/2*np.array(walls)

def drone_colors():
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

def init_colors(lines, colors, isdrone, random=np.random):
    texwidths = resolutions(lines)
    starts = texwidths.cumsum() - texwidths

    indices = np.full(texwidths.sum(), 0)
    indices[starts] = 1
    indices = np.cumsum(indices) - 1
    textures = common.gamma_decode(colors[indices])

    # Gives walls an even pattern that makes depth perception easy
    pattern = wall_pattern(textures.shape[0], random=random)
    pattern[isdrone[indices]] = 1.
    textures = textures*pattern[:, None]

    return textures, texwidths

@torch.no_grad()
def init_scene(cuda, designs, random=np.random): 
    n_drones = designs[0].n_drones

    dronelines = np.tile(drone_frame(), (n_drones, 1, 1))
    dronecolors = np.tile(drone_colors(), (n_drones, 1))

    data = dotdict(frame=drone_frame())
    lights, lightwidths, lines, linewidths, colors, isdrone  = [], [], [], [], [], []
    for d in designs:
        lights.extend([d.lights])
        lightwidths.append(len(d.lights))

        lines.extend([dronelines, d.walls])
        linewidths.append(len(dronelines) + len(d.walls))
        isdrone.extend([np.full(len(dronelines), True), np.full(len(d.walls), False)])

        colors.extend([dronecolors, d.colors])
    data['lights'], data['lightwidths'] = np.concatenate(lights), np.array(lightwidths)
    data['lines'], data['linewidths'] = np.concatenate(lines), np.array(linewidths)
    colors = np.concatenate(colors) 
    isdrone = np.concatenate(isdrone)

    data['textures'], data['texwidths'] = init_colors(data['lines'], colors, isdrone, random)

    scene = cuda.Scene(**{k: tensorify(v) for k, v in data.items()})
    cuda.bake(scene, n_drones)

    return scene
    

