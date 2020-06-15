from io import BytesIO
import logging
import requests
from tqdm.auto import tqdm
from zipfile import ZipFile
import pandas as pd
from pathlib import Path
from IPython.display import HTML
import gzip
import numpy as np
from . import tools, geometry
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from bs4 import BeautifulSoup
from rebar import parallel

log = logging.getLogger(__name__)

def download(url):
    bs = BytesIO()
    log.info(f'Downloading {url}')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers['Content-Length']) if 'Content-Length' in r.headers else None
        with tqdm(total=total, unit_scale=True, unit_divisor=1024, unit='B') as pbar:
            for chunk in r.iter_content(chunk_size=2**20): 
                pbar.update(len(chunk))
                bs.write(chunk)
    return bs.getvalue()

def cubicasa5k():
    p = Path('.cache/cubicasa.zip')
    if not p.exists():
        url = 'https://zenodo.org/record/2613548/files/cubicasa5k.zip?download=1'
        p.parent.mkdir(exist_ok=True, parents=True)
        p.write_bytes(download(url))
    return str(p)

def svgdata(regenerate=False):
    p = Path('.cache/cubicasa-svgs.json.gzip')
    if not p.exists():
        p.parent.mkdir(exist_ok=True, parents=True)
        if regenerate:
            log.info('Regenerating SVG cache from cubicasa dataset. This will require a 5G download.')
            with ZipFile(cubicasa5k()) as zf:
                pattern = r'cubicasa5k/(?P<category>[^/]*)/(?P<id>\d+)/(?P<filename>[^.]*)\.svg'
                svgs = (pd.Series(zf.namelist(), name='path')
                            .to_frame()
                            .loc[lambda df: df.path.str.match(pattern)]
                            .reset_index(drop=True))
                svgs = pd.concat([svgs, svgs.path.str.extract(pattern)], axis=1)
                svgs['svg'] = svgs.path.apply(lambda p: zf.read(p).decode())
                compressed = gzip.compress(svgs.to_json().encode())
                p.write_bytes(compressed)
        else:
            #TODO: Shift this to Github 
            url = 'https://www.dropbox.com/s/iblduqobhqomz4g/cubicasa-svgs.json.gzip?raw=1'
            p.write_bytes(download(url))
    return pd.read_json(gzip.decompress(p.read_bytes()))

def flatten(tree):
    flat = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            for kk, vv in flatten(v).items():
                flat[f'{k}/{kk}'] = vv
        else:
            flat[k] = v
    return flat

def safe_geometry(id, svg):
    try: 
        return geometry.geometry(svg)
    except:
        log.info(f'Geometry generation failed on on #{id}')

def geometrydata(regenerate=False):
    p = Path('.cache/cubicasa-geometry.zip')
    if not p.exists():
        p.parent.mkdir(exist_ok=True, parents=True)
        if True:
            log.info('Regenerating geometry cache from SVG cache.')
            with parallel.parallel(safe_geometry) as p:
                gs = p.wait({row.id: p(row.id, row.svg) for _, row in svgdata(regenerate).iterrows()})

            bs = BytesIO()
            with ZipFile(bs, 'w') as zf:
                for id, g in gs.items():
                    if g is None:
                        continue

                    for k, v in flatten(g).items():
                        vbs = BytesIO()
                        np.save(vbs, v)
                        zf.writestr(f'{id}/{k}.npy', vbs.getvalue())
            p.write_bytes(bs.getvalue())
        else:
            url = ''
            p.write_bytes(download(url))
    return str(p)
