import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import rasterio.features
from shapely.ops import cascaded_union
from shapely.geometry import Polygon, LineString
from bs4 import BeautifulSoup

MARGIN = 1.
RES = .2
SCALE = 100

from itertools import islice, cycle

def cyclic_pairs(xs):
    """Returns pairs ``(xs[i], xs[i+1])``, wrapping the last pair round to the start."""
    ys = islice(cycle(xs), 1, None)
    return list(zip(xs, ys))

def signed_area(points):
    area = 0
    for x, y in cyclic_pairs(points):
        area += x[0]*y[1] - x[1]*y[0]
    return area

def polypoints(svgpoly):
    return np.array([list(map(float, p.split(','))) for p in svgpoly.attrs['points'].split()])

def orient(points):
    if signed_area(points) > 0:
        return points
    else:
        return points[::-1]

def unique(walls):
    """Eliminate walls that are copies of other walls"""
    forward  = ((walls[:, None, :, :] - walls[None, :, ::+1, :])**2).sum(-1).sum(-1)**.5
    backward = ((walls[:, None, :, :] - walls[None, :, ::-1, :])**2).sum(-1).sum(-1)**.5
    mask = (forward < 1e-3) | (backward < 1e-3)
    mask[np.triu_indices_from(mask)] = False
    return walls[~mask.any(1)]

def svg_walls(soup):
    walls = cascaded_union([Polygon(polypoints(wall)) for wall in soup.select('.Wall>polygon')])
    doors = cascaded_union([Polygon(polypoints(door)) for door in soup.select('.Door>polygon')])

    # Sometimes the doors are slightly misaligned with the walls; dilating the doors handles this
    skeleton = walls - doors.buffer(5)
    if skeleton.geom_type == 'MultiPolygon':
        tops = [orient(np.array(g.exterior.coords)) for g in skeleton.geoms]
    elif skeleton.geom_type == 'Polygon':
        tops = [orient(np.array(skeleton.exterior.coords))]
    walls = np.concatenate([cyclic_pairs(t) for t in tops])

    # Zero-length walls cause various warnings upstream. May as well get rid of them now. 
    lengths = ((walls[:, 0] - walls[:, 1])**2).sum(1)**.5
    return unique(walls[lengths > 0])

def svg_spaces(soup):
    return [polypoints(poly) for poly in soup.select('.Space>polygon')]

def transform(walls, spaces):
    """svg coords are in centimeters from the (left, top) corner, 
    while we want metres from the (left, bottom) corner"""
    joint = np.concatenate([np.concatenate(walls), np.concatenate(spaces)])
    (left, _), (_, bot) = joint.min(0), joint.max(0)

    def tr(ps):
        x, y = ps[..., 0], ps[..., 1]
        return np.stack([x - left, bot - y], -1)/SCALE + MARGIN
    
    return tr(walls), [tr(s) for s in spaces]

def mask_transform(*args):
    points = np.concatenate([np.concatenate(a) for a in args])
    assert np.concatenate(points).min() > 0, 'Masker currently requires the points to be in the top-right quadrant'
    r, t = points.max(0) + MARGIN
    h, w = int(t/RES)+1, int(r/RES)+1
    return rasterio.transform.Affine(RES, 0, 0, 0, -RES, h*RES), (h, w)

def masks(walls, spaces, res=RES):
    """Generates a masking array from an array of walls and a list of spaces.
    
    :param walls: A (n_walls, 2, 2)-array giving the coordinates of the walls' endpoints.
    :param spaces: A list of spaces, each given as a coordinate array of the space's vertices.
    :param res: The resolution of the the masking array. 
    :return: A masking array, with indices 1, 2, ... for the spaces, 0 for free space, and -1 for walls. 
    """
    transform, shape = mask_transform(walls, spaces)
    wall_shapes = [(cascaded_union([LineString(p).buffer(.01) for p in walls]), -1)]
    space_shapes = [(Polygon(p).buffer(0), i+1) for i, p in enumerate(spaces)]
    shapes = [(s, v) for (s, v) in space_shapes + wall_shapes if not s.is_empty]
    return rasterio.features.rasterize(shapes, shape, transform=transform, all_touched=True, dtype=np.int16)

def centroids(spaces):
    # Reshape needed for the case there are no lights
    return np.array([Polygon(ps).centroid.coords[0] for ps in spaces]).reshape(-1, 2)

def geometry(svg):
    soup = BeautifulSoup(svg, features='lxml')
    walls = unique(svg_walls(soup))
    spaces = svg_spaces(soup)
    walls, spaces = transform(walls, spaces)
    return dict(
        walls=walls,
        lights=centroids(spaces),
        masks=masks(walls, spaces),
        res=RES)

def centers(indices, shape, res):
    """Converts mask (i, j) indices to the (x, y) coordinates of the ``ij`` th cell's center.

    Usually the ``shape`` and ``res`` arguments for this come directly from a geoemtry dotdict.
    
    :param indices: A (..., 2) array of indices into a masking array.
    :param shape: A tuple-like giving the height and width of the masking array.
    :param res: The resolution of the masking array.
    :return: A (..., 2) array of (x, y) coordinates
    """
    i, j = indices[..., 0] + .5, indices[..., 1] + .5
    xy = res*np.stack([j, shape[0] - i], -1)
    return xy

def indices(coords, shape, res):
    """Converts (x, y) coordinates to the (i, j) indices of the containing cell.
    
    Usually the ``shape`` and ``res`` arguments for this come directly from a geoemtry dotdict.
    
    :param indices: A (..., 2) array of (x, y) coordinates.
    :param shape: A tuple-like giving the height and width of the masking array.
    :param res: The resolution of the masking array.
    :return: A (..., 2) array of integer (i, j) indices.
    """
    x, y = coords[..., 0], coords[..., 1]
    i = (shape[0] - y/res).clip(0, shape[0]-1)
    j = (x/res).clip(0, shape[1]-1)
    return np.stack([i, j], -1).astype(int)

def display(g):
    """Visualize a geometry using matplotlib. 
    
    Supports visualizing partial geometries, that only have a subset of id/masks/walls/lights"""
    fig, ax = plt.subplots()
    ax.set_aspect(1)

    if 'id' in g:
        ax.set_title(g.id)

    if 'walls' in g:
        color = 'k' if 'masks' in g else 'k'
        lines = mpl.collections.LineCollection(g.walls, color=color, linewidth=2)
        ax.add_collection(lines)
        ax.autoscale()

    if 'lights' in g:
        for light in g.lights:
            ax.add_patch(mpl.patches.Circle(light[:2], radius=.05, color='yellow'))
        ax.autoscale()

    if 'masks' in g:
        height, width = g.res*np.array(g.masks.shape)
        extent = (0, width, 0, height)
        cm = ax.imshow(g.masks, extent=extent, cmap='tab20')
        ticks = np.arange(g.masks.min(), g.masks.max()+1)
        plt.colorbar(cm, values=ticks, ticks=ticks)
    
    return fig