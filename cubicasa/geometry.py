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
    return rasterio.transform.Affine(r/w, 0, 0, 0, -t/h, t), (h, w)

def mask_array(points, transform, shape):
    return rasterio.features.rasterize(
                        [cascaded_union(points)], 
                        out_shape=shape, 
                        transform=transform).astype(np.bool)

def masks(walls, spaces):
    transform, shape = mask_transform(walls, spaces)
    return dict(
        walls=mask_array([LineString(p).buffer(RES) for p in walls], transform, shape),
        spaces=mask_array([Polygon(p).buffer(0) for p in spaces], transform, shape),
        res=RES)

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
        spaces={i: s for i, s in enumerate(spaces)},
        centroids=centroids(spaces),
        masks=masks(walls, spaces))
