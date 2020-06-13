import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, islice, combinations
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import cascaded_union
from collections import namedtuple
import scipy as sp
import scipy.ndimage
import logging
from rebar import dotdict

logging.getLogger('rasterio.env').setLevel('WARN')
import rasterio
import rasterio.features


MIN_ZONE_RADIUS = .5
MARGIN = 1.
RES = .2

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

def normalize(v):
    return v/(v**2).sum()**.5

def perp(v):
    return np.array([v[1], -v[0]])

def cyclic_pairs(xs):
    ys = islice(cycle(xs), 1, None)
    return list(zip(xs, ys))

def names(drones):
    potential =  ['ella', 'zoe', 'liz', 'mike'] + [f'drone-{i}' for i in range(5, 100)]
    return potential[:drones]

def sample(poly):
    """Rejection sampling to pick points at random from inside a design"""
    poly = Polygon(poly) if isinstance(poly, np.ndarray) else poly
    l, b, r, t = poly.bounds
    for _ in range(1000):
        x = np.random.uniform(l, r)
        y = np.random.uniform(b, t)
        if poly.contains(Point(x, y)):
            return x, y
    else:
        raise ValueError('Couldn\'t find an internal point in the design after a thousand tries')

def distinct_positions(poly, n, sep):
    for _ in range(100):
        positions = np.array([sample(poly) for _ in range(n)])
        valid = all([((p - q)**2).sum() > sep**2 for p, q in combinations(positions, 2)])  
        if valid:
            return positions
    else:
        raise ValueError('Couldn\'t find distinct positions')

def signed_area(points):
    area = 0
    for x, y in cyclic_pairs(points):
        area += x[0]*y[1] - x[1]*y[0]
    return area

def area(points):
    return abs(signed_area(points))

def cyclic_triples(ys):
    xs = islice(cycle(ys), len(ys) - 1, None)
    zs = islice(cycle(ys), 1, None)
    return list(zip(xs, ys, zs))

def erode(points, dist=.25):
    return Polygon(points).buffer(-dist)

def subzones(n_drones, centers, radii, lowers=None, uppers=None, subradii=1.):
    centers = np.array(centers)
    radii = np.array(radii)

    # This is a regular circle packing. It's optimal up until 6 drones:
    # https://en.wikipedia.org/wiki/Circle_packing_in_a_circle
    if n_drones > 1:
        st = np.sin(np.pi/n_drones) 
        offset_radius = radii*1/(1 + st)
        sub_radius = radii*subradii*st/(1 + st)
    else:
        offset_radius, sub_radius = radii*0., radii*subradii

    angles = np.linspace(0, 2*np.pi, n_drones, endpoint=False)
    offsets = offset_radius[:, None, None]*np.stack([np.cos(angles), np.sin(angles)]).T[None, :, :]
    new_centers = offsets + centers[:, None, :]
    new_radii = np.repeat(sub_radius[:, None], n_drones, 1)

    if (lowers is None) and (uppers is None):
        if n_drones > 1:
            # Drones that point at eachother
            mids = np.repeat((180/np.pi*angles + 180)[None, :], len(centers), 0)
            new_lowers, new_uppers = mids, mids
        else:
            new_lowers, new_uppers = np.full_like(new_radii, -179), np.full_like(new_radii, +179)
    elif (lowers is not None) and (uppers is not None):
        new_lowers, new_uppers = np.array(lowers), np.array(uppers)
        new_lowers, new_uppers = np.repeat(new_lowers[:, None], n_drones, 1), np.repeat(new_uppers[:, None], n_drones, 1)
    else:
        raise ValueError('Either both of lowers/uppers must be None, or neither are')

    return dotdict(centers=new_centers, radii=new_radii, lowers=new_lowers, uppers=new_uppers)

class Mask:

    def __init__(self, f, points, top_right=None, res=RES):
        assert np.concatenate(points).min() > 0, 'Design currently needs to be in the top-right quadrant'
        top_right = np.concatenate(points).max(0) + MARGIN if top_right is None else top_right
        r, t = top_right
        
        self.res = res
        self.h, self.w = int(t/res), int(r/res)
        self.transform = rasterio.transform.Affine(r/self.w, 0, 0, 0, -t/self.h, t)

        polys = cascaded_union([f(ps) for ps in points])
        self.values = rasterio.features.rasterize(
                            [polys], 
                            out_shape=(self.h, self.w), 
                            transform=self.transform)

def mask_walls(*args, **kwargs):
    return Mask(lambda ps: LineString(ps).buffer(RES), *args, **kwargs)

def mask_spaces(*args, **kwargs):
    return Mask(lambda ps: Polygon(ps).buffer(0), *args, **kwargs)

def masks(walls, spaces):
    top_right = np.concatenate([np.concatenate(walls), np.concatenate(spaces)]).max(0)
    return mask_walls(walls, top_right), mask_spaces(spaces, top_right)

def start_zones(wall_mask, space_mask):
    res = wall_mask.res

    indices = np.dstack(np.indices(wall_mask.values.shape))

    centers, radii = [], []
    mask = wall_mask.values.copy()
    for _ in range(100):
        d = sp.ndimage.distance_transform_edt(1 - mask)*space_mask.values
        #TODO: Use a random local max greater than the min radius rather than the global max, to get more diverse starts
        center = np.unravel_index(np.argmax(d), d.shape)
        radius = d[center] - 1
        
        centers.append(center), radii.append(radius)
        
        mask[((indices - center)**2).sum(-1)**.5 <= radius + res] = 1
        
        if radius < MIN_ZONE_RADIUS/res:
            break
    centers, radii = np.stack(centers), np.array(radii)
    radii = res*radii

    centers[:, 0] = wall_mask.h - centers[:, 0]
    centers = res*centers[:, ::-1]
    return centers, radii

class Design:
    
    def __init__(self, 
            id, centers, radii, lights, walls, 
            colormap=None, lowers=None, uppers=None, mask=None, 
            meta=dotdict()):
        self.id = id
        self.centers = np.array(centers, dtype=float)
        self.radii = np.array(radii, dtype=float)
        self.lights = np.array(lights, dtype=float)
        self.walls = np.array(walls, dtype=float)
        self.respawns = len(centers)
        self.xlim = (self.walls[..., 0].min()-MARGIN, self.walls[..., 0].max()+MARGIN)
        self.ylim = (self.walls[..., 1].min()-MARGIN, self.walls[..., 1].max()+MARGIN)
        # self.mask = mask_walls(walls) if mask is None else mask
        self.meta = meta

        if lowers is None:
            lowers = -180*np.ones_like(self.radii)
        self.lowers = np.array(lowers, dtype=float)
        if uppers is None:
            uppers = +180*np.ones_like(self.radii)
        self.uppers = np.array(uppers, dtype=float)

        colormap = np.arange(len(self.walls)) if colormap is None else colormap
        self.colors = np.array([mpl.colors.to_rgb(COLORS[i % len(COLORS)]) for i in colormap])

        self.n_drones = self.centers.shape[1]

        assert self.centers.shape[:2] == self.radii.shape
        assert self.centers.shape[:2] == self.lowers.shape
        assert self.centers.shape[:2] == self.uppers.shape
        assert len(self.colors) == len(self.walls)
        assert len(self.lights) > 0, 'Simulator can\'t currently handle zero lights'

    def plot(self, ax=None, zones=True, lights=True, **kwargs):
        ax = (ax or plt.subplot())

        for w, c in zip(self.walls, self.colors):
            ax.add_line(mpl.lines.Line2D(*w.T, **{'color': c, **kwargs}))

        if zones:
            for z in range(self.centers.shape[0]):
                for d in range(self.centers.shape[1]):
                    c, r, l, u = self.centers[z, d], self.radii[z, d], self.lowers[z, d], self.uppers[z, d]
                    r = max(r, .05)
                    d = max((u-l)/2., 1/2.)
                    ax.add_patch(mpl.patches.Circle(c, r, **{'edgecolor': 'C2', 'facecolor': 'w', 'hatch': '/', **kwargs}))
                    ax.add_patch(mpl.patches.Wedge(c, r, l, u, **{'color': 'C2', 'alpha': .2, **kwargs}))

        if lights:
            for l in self.lights:
                ax.add_patch(mpl.patches.Circle(l[:2], radius=.1, **{'color': 'C1', 'alpha': l[2], **kwargs}))
        
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect(1)

        ax.set_title(self.id)
        return ax

    def __repr__(self):
        return f'{type(self).__name__}({self.id})'

    def __str__(self):
        return repr(self)

class DesignError(RuntimeError):

    def __init__(self, message): 
        self.message = message

def wall_cycle(corners):
    return np.array(list(zip(corners, islice(cycle(corners), 1, None))))

def box_walls(x, y):
    angles = np.array([45, 135, 225, 315])
    corners = 2**.5*np.array([[np.cos(a), np.sin(a)] for a in np.pi/180*angles])
    corners = np.array([x, y])[None, :]*corners/2.
    return wall_cycle(corners)

def point_start(angle, position, n_drones=1):
    return subzones(
                n_drones=n_drones,
                centers=[position],
                radii=[0.],
                lowers=[angle],
                uppers=[angle])
