import numpy as np
from .common import box_walls, point_start, Design
            
def gallery(n_agents=1, x=20, y=2):
    assert x > 4
    walls = box_walls(x, y)
    lights = np.linspace(-x/2+2, x/2-2, int(x//4))
    lights = np.stack([lights, np.zeros_like(lights), np.ones_like(lights)], -1)
    return Design(
            id=f'gallery-{x}',
            lights=lights,
            walls=walls,
            **point_start(90., [0., 0.]))

def corridor(n_agents=1, x=20, y=2):
    assert x > 4
    walls = box_walls(x, y)
    lights = np.linspace(-x/2+2, x/2-2, int(x//4))
    lights = np.stack([lights, np.zeros_like(lights), np.ones_like(lights)], -1)
    return Design(
            id=f'corridor-{x}',
            lights=lights,
            walls=walls,
            **point_start(0., [0., 0.]))

def box(n_agents=1, x=20, y=20):
    assert x > 4
    walls = box_walls(x, y)
    light_xs = np.linspace(-x/2+2, x/2-2, int(x/min(4, x/2)))
    light_ys = np.linspace(-y/2+2, y/2-2, int(y/min(4, y/2)))
    alpha_xs = np.ones_like(light_xs[:-1])
    alpha_ys = np.ones_like(light_ys[:-1])
    lights = np.r_[
        np.c_[light_xs[:-1], np.full_like(alpha_xs, light_ys[0]), alpha_xs],
        np.c_[np.full_like(alpha_ys, light_xs[-1]), light_ys[:-1], alpha_ys],
        np.c_[light_xs[1:], np.full_like(alpha_xs, light_ys[-1]), alpha_xs],
        np.c_[np.full_like(alpha_ys, light_xs[0]), light_ys[1:], alpha_ys]]
    return Design(
            id=f'box-{x}-{y}',
            lights=lights,
            walls=walls,
            **point_start(0., [0., 0.]))

