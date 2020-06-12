import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from contextlib import contextmanager
from IPython.display import HTML, display
from bs4 import BeautifulSoup
import numpy as np
from . import common, cache, parallel
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import cascaded_union
import multiprocessing
from tqdm.auto import tqdm
import rebar

log = rebar.logger()

URL = "https://zenodo.org/record/2613548/files/cubicasa5k.zip?download=1"
CLOUD_CACHE = 'tmp/cubicasa5k.zip'

SCALE = 100

@contextmanager
def zipfile(zf=None):
    # # Caches in Google Storage first, then in a local dir
    # from ..cloud import storage
    # cache = Path('.cache/cubicasa.zip')
    # if not cache.exists():
    #     cloud_cache = storage.Path(CLOUD_CACHE)
    #     if not cloud_cache.exists():
    #         raw = cache.download(URL)
    #         cloud_cache.write_bytes_multipart(raw)
    #     cache.parent.mkdir(exist_ok=True, parents=True)
    #     cache.write_bytes(cloud_cache.read_bytes())
    if zf:
        yield zf
    else:
        c = Path('.cache/cubicasa.zip')
        if not c.exists():
            raw = cache.download(URL)
            c.parent.mkdir(exist_ok=True, parents=True)
            c.write_bytes(raw)

        with ZipFile(str(c)) as zf:
            yield zf

def files(zf=None):
    pattern = r'cubicasa5k/(?P<category>[^/]*)/(?P<id>\d+)/(?P<filename>[^.]*)\.(?P<filetype>.*)'
    with zipfile(zf) as zf:
        return (pd.Series(zf.namelist())
                    .loc[lambda s: s.str.match(pattern)]
                    .str.extract(pattern)
                    .assign(id=lambda df: pd.to_numeric(df.id)))

@cache.memcache('{split}')
def ids(split=None, zf=None):
    ids = (files(zf)
                .query('category == "high_quality_architectural" & filename == "model"')
                .id
                .sort_values()
                .reset_index(drop=True))
    # Randomize the order
    ids.values[:] = np.random.RandomState(12041955).permutation(ids.values)
    # Take 90% for the training set, 10% for the test
    if split == 'train':
        ids = ids[ids % 10 != 0]
    if split == 'test':
        ids = ids[ids % 10 == 0]
    return ids

@cache.autocache('{category}-{index}')
def svgfile(index, category='high_quality_architectural', zf=None):
    with zipfile(zf) as zf:
        id = ids()[index]
        return zf.read(f'cubicasa5k/{category}/{id}/model.svg').decode()

def _cache(count, base, period):
    with zipfile() as zf:
        for i in tqdm(np.arange(base, count, period), desc=f'base-{base}'):
            svgfile(i, zf=zf)

def bounds(soup):
    return np.array([float(c) for c in soup.select_one('svg').attrs['viewbox'].split()])

def transformation(soup):
    l, b, r, t = bounds(soup)/SCALE
    origin = np.array([l - common.MARGIN, b - common.MARGIN])
    
    def transform(points):
        if np.isscalar(points):
            return points/SCALE
        elif isinstance(points, list):
            return [transform(p) for p in points]
        else:
            return points/SCALE - origin
    
    return transform

def polypoints(poly):
    return np.array([list(map(float, p.split(','))) for p in poly.attrs['points'].split()])

def orient(points):
    if common.signed_area(points) > 0:
        return points
    else:
        return points[::-1]

def walltops(soup):
    walls = []
    for wall in soup.select('.Wall'):
        wallpoints = polypoints(wall.polygon)
        doorspolys = [Polygon(polypoints(poly)) for poly in wall.select('.Door>polygon')]
        if doorspolys:
            wallpoly, doorspoly = Polygon(wallpoints), cascaded_union(doorspolys)
            nondoor = wallpoly - doorspoly
            if nondoor.geom_type != 'MultiPolygon':
                # If it's still a single poly, the door is misaligned. Can dilate the door by a few cm to fix that. 
                nondoor = wallpoly - doorspoly.buffer(.15)
            walls.extend([orient(np.array(g.exterior.coords)) for g in nondoor.geoms])
        else:
            walls.append(orient(wallpoints))
    return walls
            
def raw_walls(soup):
    walls = np.concatenate([common.cyclic_pairs(wt) for wt in walltops(soup)])
    # Zero-length walls cause various warnings upstream. May as well get rid of them now. 
    lengths = ((walls[:, 0] - walls[:, 1])**2).sum(1)**.5
    return walls[lengths > 0]

def raw_spaces(soup):
    return [polypoints(poly) for poly in soup.select('.Space>polygon')]

def plot(svg):
    return display(HTML(svg))

def unique(walls):
    """Eliminate walls that are copies of other walls"""
    forward  = ((walls[:, None, :, :] - walls[None, :, ::+1, :])**2).sum(-1).sum(-1)**.5
    backward = ((walls[:, None, :, :] - walls[None, :, ::-1, :])**2).sum(-1).sum(-1)**.5
    mask = (forward < 1e-3) | (backward < 1e-3)
    mask[np.triu_indices_from(mask)] = False
    return walls[~mask.any(1)]

def lights(spaces):
    # Reshape needed for the case there are no lights
    coords = np.array([Polygon(ps).centroid.coords[0] for ps in spaces]).reshape(-1, 2)
    intensities = .5 + np.random.rand(len(coords), 1)
    return np.concatenate([coords, intensities], 1)

def colormap(walls, spaces):
    sps = [Polygon(sp) for sp in spaces]
    ds = []
    for l in walls:
        l = LineString(l)
        ds.append([l.distance(sp) for sp in sps])
    closest = np.array(ds)
    if closest.size: 
        return closest.argmin(1)
    else:
        return np.zeros(len(walls), dtype=int)

@cache.autocache('{index}-{n_drones}')
def _cubicasa(index, n_drones, error=False):
    """This will fail on about 10% of designs because my generation scheme isn't totally robust

    Lights: 0-125
    Lines: 8-1250
    Texels: 1k-750k
    """
    if error:
        raise ValueError('You\'ll want to call cubicache first')
    try:
        soup = BeautifulSoup(svgfile(index), 'lxml')
        transform = transformation(soup)
        walls = unique(transform(raw_walls(soup)))
        spaces = transform(raw_spaces(soup))

        wall_mask, space_mask = common.masks(walls, spaces)
        centers, radii = common.start_zones(wall_mask, space_mask)

        return common.Design(
                    id=f'cubicasa-{index}',
                    lights=lights(spaces),
                    walls=walls,
                    colormap=colormap(walls, spaces),
                    mask=wall_mask,
                    **common.subzones(
                        n_drones=n_drones,
                        centers=centers,
                        radii=radii))
    except Exception as e:
        log.warn(f'Failed to create design for index #{index}: {e}')

def cubicasa(n_designs=1, n_drones=1, split='train', rank=0):
    designs = []
    options = ids(split=split).index
    current = 0
    start = rank*n_designs
    while len(designs) < start + n_designs:
        d = _cubicasa(options[current], n_drones, error=True)
        if d:
            designs.append(d)
        current += 1
        if current >= len(options):
            raise ValueError(f'Too many designs requested - max available is {len(designs)}')
    return designs[start:]

def n_cubicasa():
    return len(ids())

def cubicache(n_drones):
    with parallel.parallel(_cubicasa) as p:
        p.wait([p(i, n_drones, error=False) for i in range(n_cubicasa())])
