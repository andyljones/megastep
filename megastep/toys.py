import numpy as np
from . import geometry
from rebar import arrdict

def box(width=5):
    """A :ref:`geometry <geometry>` which is just a simple box, with one room and one light inside it."""
    corners = [(np.cos(t), np.sin(t)) for t in np.arange(np.pi/4, 2*np.pi, np.pi/2)]
    corners = width/2**.5 *np.array(corners) + width/2 + geometry.MARGIN
    walls = np.stack(geometry.cyclic_pairs(corners))
    spaces = [corners]

    return arrdict.arrdict(
        walls=walls,
        lights=np.full((1, 2), width/2 + geometry.MARGIN),
        masks=geometry.masks(walls, spaces),
        res=geometry.RES)

def column(width=5, column_width=.1):
    """A :ref:`geometry <geometry>` which is just a simple 'column' (aka small box), with one room around it"""
    corners = [(np.cos(t), np.sin(t)) for t in np.arange(np.pi/4, 2*np.pi, np.pi/2)]
    column_corners = column_width/2**.5 *np.array(corners) + width/2 + geometry.MARGIN
    walls = np.stack(geometry.cyclic_pairs(column_corners))
    spaces = [width/2**.5 * np.array(corners) + width/2 + geometry.MARGIN]

    return arrdict.arrdict(
        walls=walls,
        lights=2**.5 * np.array(corners) + width/2 + geometry.MARGIN,
        masks=geometry.masks(walls, spaces),
        res=geometry.RES)
