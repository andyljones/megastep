import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from matplotlib.animation import FuncAnimation
from . import common
from IPython.display import display

VIEW_RADIUS = 5

def plot_images(axes, arrs):
    ims = imshow_arrays(arrs)
    D = len(ims)
    H, W = ims[0].shape[:2]
    
    aspect = 2*W/(D*H)  # TODO: Reverse this
    for a in range(D):
        ax = axes[a]
        ax.imshow(ims[a], aspect=aspect)
        ax.set_yticks(np.arange(H))
        ax.set_ylim(H-.5, -.5)
        ax.set_yticklabels(arrs.keys())
        ax.set_xticks([])
        ax.set_title(f'agent #{a}', fontdict={'color': f'C{a}', 'weight': 'bold'})

def n_agent_texels(state):
    D = state.agents.angles.shape[0]
    F = len(state.scene.frame)
    return state.scene.widths[:D*F].sum()

def line_arrays(state):
    scene = state.scene

    starts = scene.widths.cumsum() - scene.widths
    tex_starts = np.zeros(len(scene.textures), dtype=int)
    tex_starts[starts] = 1
    tex_starts = tex_starts.cumsum() - 1
    tex_offsets = np.arange(len(tex_starts)) - starts[tex_starts]

    seg_starts = tex_offsets/scene.widths[tex_starts]
    seg_starts = scene.lines[tex_starts, 0]*(1 - seg_starts[:, None]) + scene.lines[tex_starts, 1]*seg_starts[:, None]

    seg_ends = (tex_offsets+1)/scene.widths[tex_starts]
    seg_ends = scene.lines[tex_starts, 0]*(1 - seg_ends[:, None]) + scene.lines[tex_starts, 1]*seg_ends[:, None]

    lines = np.stack([seg_starts, seg_ends]).transpose(1, 0, 2)

    baked = scene.baked.copy()
    baked[:n_agent_texels(state)] = 1.

    colors = common.gamma_encode(scene.textures*baked[:, None])
    return lines, colors

def plot_lights(diagram, state):
    for light in state.scene.lights:
        diagram.add_patch(mpl.patches.Circle(light[:2], radius=.05, alpha=light[2], color='yellow'))

def extent(state, cull, radius=VIEW_RADIUS):
    if cull:
        r, t = state.agents.positions.max(0) + radius 
        l, b = state.agents.positions.min(0) - radius
    else:
        r, t = state.scene.lines.max(0).max(0) + 1
        l, b = state.scene.lines.min(0).min(0) - 1

    w = max(t - b, r - l)/2
    cx, cy = (r + l)/2, (t + b)/2

    return (cx-w, cx+w), (cy-w, cy+w)

def plot_lines(diagram, state, cull=True):
    lines, colors = line_arrays(state)

    (l, r), (b, t) = extent(state, cull)
    xs, ys = lines[:, :, 0], lines[:, :, 1]
    mask = (l < xs) & (xs < r) & (b < ys) & (ys < t)
    mask = mask.any(-1)

    seen = mpl.collections.LineCollection(lines[mask], colors=colors[mask], linestyle='solid', linewidth=2)
    diagram.add_collection(seen)

def adjust_view(diagram, state, cull=True):
    xs, ys = extent(state, cull)

    diagram.set_xlim(*xs)
    diagram.set_ylim(*ys)

    diagram.set_aspect(1)
    diagram.set_facecolor('#c6c1b3')

def plot_wedge(ax, pose, distance, fov, radians=False, **kwargs):
    scale = 180/np.pi if radians else 1
    left = scale*pose.angles - fov/2
    right = scale*pose.angles + fov/2
    width = distance - common.AGENT_RADIUS 
    wedge = mpl.patches.Wedge(
                    pose.positions, distance, left, right, width=width, **kwargs)
    ax.add_patch(wedge)

def plot_fov(diagram, state, distance=1, field='agents'):
    a = len(getattr(state, field).angles)
    for i in range(a):
        plot_wedge(diagram, getattr(state, field)[i], distance, state.fov, color=f'C{i}', alpha=.1)

def plot_poses(poses, ax=None, radians=True, color='C9', **kwargs):
    """Not used directly here, but often useful for code using this module"""
    ax = ax or plt.subplot()
    for angle, position in zip(poses.angles, poses.positions):
        ax.add_patch(mpl.patches.Circle(position, radius=common.AGENT_RADIUS, edgecolor=color, facecolor=[0,0,0,0]))

        scale = 1 if radians else np.pi/180
        offset = common.AGENT_RADIUS*np.array([np.cos(scale*angle), np.sin(scale*angle)])
        line = np.stack([position, position + offset])
        ax.plot(*line.T, color=color)
    return ax

def plot(state, fig=None):
    fig = fig or plt.figure()
    diagram = fig.add_axes([0, 0, 1, 1])

    plot_lights(diagram, state)
    plot_lines(diagram, state, cull=False)
    plot_fov(diagram, state, 1000) # suitably large number
    adjust_view(diagram, state, cull=False)

    for ax in [diagram]:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig