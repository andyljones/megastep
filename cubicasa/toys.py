import numpy as np
from . import geometry
from rebar import arrdict

def box(width=5):
    corners = [(np.cos(t), np.sin(t)) for t in np.arange(np.pi/4, 2*np.pi, np.pi/2)]
    corners = width/2**.5 *np.array(corners) + width/2 + geometry.MARGIN
    walls = np.stack(geometry.cyclic_pairs(corners))
    spaces = [corners]

    return arrdict(
        walls=walls,
        centroids=np.full((1, 2), width/2 + geometry.MARGIN),
        masks=geometry.masks(walls, spaces),
        res=geometry.RES)