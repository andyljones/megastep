from io import BytesIO
import logging
import requests
from tqdm.auto import tqdm
from zipfile import ZipFile
import pandas as pd
from pathlib import Path
import gzip
import numpy as np
from rebar import parallel, dotdict
import ast
from pkg_resources import resource_filename
from pathlib import Path

log = logging.getLogger(__name__)

LICENSE = """The cubicasa geometry is based on the Cubicasa5k dataset. 
While megastep as a whole can be used for any purpose, this geometry 
specifically is offered for non-commercial use only. You can read 
more about this - and about alternatives - in the megastep FAQ at 

http://andyljones.com/megastep/faq
"""

REJECTION = """
You entered "{i}". You cannot download the cubicasa dataset 
without confirming you understand the licensing conditions. Please read 
the FAQ for suggestions about alternative geometries. 

If your problem is specifically entering the 'Y' character, you can run 
this snippet from a console:
```
python -c "from megastep import cubicasa; cubicasa.force_confirm()
```
"""

PATH = Path(resource_filename(__package__, '.cubicasa-confirmed'))

def confirm():
    if PATH.exists():
        return 

    print(LICENSE)
    print('Please enter "Y" to confirm you understand this.\n', flush=True)
    i = input('[Y/N]: ')
    if i in 'yY':
        PATH.touch()
        print('\nConfirmed.')
        return 
    else:
        print(REJECTION.format(i=i))
        raise ValueError('You refused to confirm that you understand the license restrictions.')

def force_confirm():
    """:ref:`As described in the FAQ <cubicasa-license>, the cubicasa dataset has a non-commercial use restriction`. 
    
    Most users will be prompted for their understanding of this restriction when they first download the dataset, but
    this might cause trouble for people who are using automated systems, so calling this function 
    is offered as an alternative.
    """
    print(LICENSE)
    print('By calling `force_confirm`, you confirm you understand this.')
    PATH.touch()

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

def svg_data(regenerate=False):
    p = Path('.cache/cubicasa-svgs.json.gz')
    if not p.exists() or regenerate:
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

def unflatten(d):
    tree = type(d)()
    for k, v in d.items():
        parts = k.split('/')
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, type(d)())
        node[parts[-1]] = v
    return tree
        
def safe_geometry(id, svg):
    try: 
        # Hide the import since it uses a fair number of libraries not used elsewhere.
        from . import geometry
        return geometry.geometry(svg)
    except:
        # We'll lose ~8 SVGs to them not having any spaces
        log.info(f'Geometry generation failed on on #{id}')

def fastload(raw):
    """Most of the time in np.load is spent parsing the header, since it could have a giant mess of record types in
    it. But we know here that it doesn't! So we can do the parsing with direct slices into the byte array, and skip a
    bunch of checks.
    
    Can push x3 faster than this by writing a regex for the descr and shape, but that's optimizing a bit too hard for 
    our purposes.
    
    Credit to @pag for pointing this out to me once upon a time"""
    headerlen = np.frombuffer(raw[8:9], dtype=np.uint8)[0]
    header = ast.literal_eval(raw[10:10+headerlen].decode())
    return np.frombuffer(raw[10+headerlen:], dtype=header['descr']).reshape(header['shape'])

def geometry_data(regenerate=False):
    # Why .npz.gz? Because applying gzip manually manages x10 better compression than
    # np.savez_compressed. They use the same compression alg, so I assume the difference
    # is in the default compression setting - which isn't accessible in np.savez_compressec.
    p = Path('.cache/cubicasa-geometry.npz.gz')
    if not p.exists() or regenerate:
        p.parent.mkdir(exist_ok=True, parents=True)
        if regenerate:
            log.info('Regenerating geometry cache from SVG cache.')
            with parallel.parallel(safe_geometry) as pool:
                gs = pool.wait({str(row.id): pool(row.id, row.svg) for _, row in svg_data().iterrows()})
            gs = flatten({k: v for k, v in gs.items() if v is not None})

            bs = BytesIO()
            np.savez(bs, **gs)
            p.write_bytes(gzip.compress(bs.getvalue()))
        else:
            #TODO: Shift this to Github 
            url = 'https://www.dropbox.com/s/3ohut8lvmr8lkwg/cubicasa-geometry.npz.gz?raw=1'
            p.write_bytes(download(url))

    # np.load is kinda slow. 
    raw = gzip.decompress(p.read_bytes())
    with ZipFile(BytesIO(raw)) as zf:
        flat = dotdict.dotdict({n[:-4]: fastload(zf.read(n)) for n in zf.namelist()})
    return unflatten(flat)

_cache = None
def sample(n_geometries, split='training', seed=1):
    """Returns a random sample of cubicasa :ref:`geometries <geometry>`. 

    If you pass the same arguments, you'll get the same sample every time.

    There are 4,992 unique geometries, split into a 4,492-geometry training set and a 500-geometry test set.

    **Caching**

    The geometries are derived from the `Cubicasa5k <https://github.com/CubiCasa/CubiCasa5k>`_ dataset.

    The first time you call this function, it'll fetch and cache a ~10MB precomputed geometries file. This is far
    easier to work with than the full 5GB Cuibcasa5k dataset. If you want to recompute the geometries from scratch
    however, import this module and try calling :: 
    
        svg_data(regenerate=True) 
        geometry_data(regenerate=True)

    **Parameters**

    :param n_designs: The number of geometries to return
    :type n_designs: int
    :param split: Whether to return a sample from the ``training`` set, the ``test`` set, or ``all`` . The split is
        90/10 in favour of the training set. Defaults to ``training`` . 
    :type split: str
    :param seed: The seed to use when allocating the training and test sets.

    :return: A list of geometries.
    """
    confirm()
    global _cache
    if _cache is None:
        _cache = geometry_data()
        # Add the ID, since we're going to return this as a list
        _cache = type(_cache)({k: type(v)({'id': k, **v}) for k, v in _cache.items()})
    
    cutoff = int(.9*len(_cache))
    order = np.random.RandomState(seed).permutation(sorted(_cache))
    if split == 'training':
        order = order[:cutoff]
    elif split == 'test':
        order = order[cutoff:]
    elif split == 'all':
        order = order
    else:
        raise ValueError('Split must be train/test/all')

    return [_cache[order[i % len(order)]] for i in range(n_geometries)]
