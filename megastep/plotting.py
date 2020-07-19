"""TODO-DOCS Plotting docs"""
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from . import core
from rebar import arrdict, dotdict
from matplotlib import tight_bbox

VIEW_RADIUS = 5

def imshow_arrays(arrs, transpose=False):
    """Args:
        arrs: `{name: A x C x H x W}`
    """
    arrs = {k: v.transpose(0, 3, 1, 2) if transpose else v for k, v in arrs.items()}
    [A] = {v.shape[0] for v in arrs.values()}
    ims = {}
    for a in range(A):
        layers = []
        for k, v in arrs.items():
            layer = v[a].astype(float)
            if layer.shape[0] == 1:
                layer = layer.repeat(3, 0)
            else:
                layer = core.gamma_encode(layer)
            layers.append(layer)
        layers = np.concatenate(layers, 1)
        ims[a] = layers.transpose(1, 2, 0)
    return ims

def plot_images(arrs, axes=None, aspect=1, **kwargs):
    ims = imshow_arrays(arrs, **kwargs)
    A = len(ims)
    H, W = ims[0].shape[:2]

    axes = plt.subplots(A, 1, squeeze=False)[1].flatten() if axes is None else axes
    
    # Aspect is height/width
    aspect = aspect/min(A, 4)*W/H
    for a in range(A):
        ax = axes[a]
        ax.imshow(ims[a], aspect=aspect, interpolation='none')
        ax.set_yticks(np.arange(H))
        ax.set_ylim(H-.5, -.5)
        ax.set_yticklabels(arrs.keys())
        ax.set_xticks([])
        ax.set_title(f'agent #{a}', fontdict={'color': f'C{a}', 'weight': 'bold'})

    return axes

def n_agent_texels(scenery):
    A = scenery.n_agents
    M = len(scenery.model)
    return scenery.textures.widths[:A*M].sum()

def line_arrays(state):
    scenery = state.scenery

    starts = scenery.textures.starts
    tex_starts = np.zeros(len(scenery.textures.vals), dtype=int)
    tex_starts[starts] = 1
    tex_starts = tex_starts.cumsum() - 1
    tex_offsets = np.arange(len(tex_starts)) - starts[tex_starts]

    seg_starts = tex_offsets/scenery.textures.widths[tex_starts]
    seg_starts = scenery.lines[tex_starts, 0]*(1 - seg_starts[:, None]) + scenery.lines[tex_starts, 1]*seg_starts[:, None]

    seg_ends = (tex_offsets+1)/scenery.textures.widths[tex_starts]
    seg_ends = scenery.lines[tex_starts, 0]*(1 - seg_ends[:, None]) + scenery.lines[tex_starts, 1]*seg_ends[:, None]

    lines = np.stack([seg_starts, seg_ends]).transpose(1, 0, 2)

    baked = scenery.baked.vals.copy()
    baked[:n_agent_texels(scenery)] = 1.

    colors = core.gamma_encode(scenery.textures.vals*baked[:, None])
    return lines, colors

def plot_lights(ax, state):
    vmin = state.scenery.lights[:, 2].min() - 1e-2
    vmax = state.scenery.lights[:, 2].max()
    for light in state.scenery.lights:
        alpha = (light[2] - vmin)/(vmax - vmin)
        ax.add_patch(mpl.patches.Circle(light[:2], radius=.05, alpha=alpha, color='yellow'))

def extent(state, zoom, radius=VIEW_RADIUS):
    if zoom:
        r, t = state.agents.positions.max(0) + radius 
        l, b = state.agents.positions.min(0) - radius
    else:
        r, t = state.scenery.lines.max(0).max(0) + 1
        l, b = state.scenery.lines.min(0).min(0) - 1

    w = max(t - b, r - l)/2
    cx, cy = (r + l)/2, (t + b)/2

    return (cx-w, cx+w), (cy-w, cy+w)

def plot_lines(ax, state, zoom=True):
    lines, colors = line_arrays(state)

    (l, r), (b, t) = extent(state, zoom)
    xs, ys = lines[:, :, 0], lines[:, :, 1]
    mask = (l < xs) & (xs < r) & (b < ys) & (ys < t)
    mask = mask.any(-1)

    seen = mpl.collections.LineCollection(lines[mask], colors=colors[mask], linestyle='solid', linewidth=2)
    ax.add_collection(seen)

def adjust_view(ax, state, zoom=True):
    xs, ys = extent(state, zoom)

    ax.set_xlim(*xs)
    ax.set_ylim(*ys)

    ax.set_aspect(1)
    ax.set_facecolor('#c6c1b3')

def plot_wedge(ax, pose, distance, fov, radians=False, **kwargs):
    scale = 180/np.pi if radians else 1
    left = scale*pose.angles - fov/2
    right = scale*pose.angles + fov/2
    width = distance - core.AGENT_RADIUS 
    wedge = mpl.patches.Wedge(
                    pose.positions, distance, left, right, width=width, **kwargs)
    ax.add_patch(wedge)

def plot_fov(ax, state, distance=1, field='agents'):
    a = len(getattr(state, field).angles)
    for i in range(a):
        plot_wedge(ax, getattr(state, field)[i], distance, state.fov, color=f'C{i}', alpha=.1)

def plot_poses(poses, ax=None, radians=True, color='C9', **kwargs):
    """Not used directly here, but often useful for code using this module"""
    ax = ax or plt.subplot()
    for angle, position in zip(poses.angles, poses.positions):
        ax.add_patch(mpl.patches.Circle(position, radius=core.AGENT_RADIUS, edgecolor=color, facecolor=[0,0,0,0]))

        scale = 1 if radians else np.pi/180
        offset = core.AGENT_RADIUS*np.array([np.cos(scale*angle), np.sin(scale*angle)])
        line = np.stack([position, position + offset])
        ax.plot(*line.T, color=color)
    return ax
