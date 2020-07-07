import os
from pathlib import Path
import shutil
import multiprocessing as mp
import pandas as pd
import re
from . import dotdict

ROOT = 'output/traces'

def resolve(run_name):
    if isinstance(run_name, str):
        return run_name
    if isinstance(run_name, int):
        times = {p: p.stat().st_ctime for p in Path(ROOT).iterdir()}
        paths = sorted(times, key=times.__getitem__)
        return paths[run_name].parts[-1]
    raise ValueError(f'Can\'t find a run corresponding to {run_name}')

def run_dir(run_name):
    run_name = resolve(run_name)
    return Path(ROOT) / run_name

def subdirectory(run_name, group, channel=''):
    if channel:
        return run_dir(run_name) / group / channel
    else:
        return run_dir(run_name) / group 

def clear(run_name, group=None):
    if group is None:
        shutil.rmtree(run_dir(run_name), ignore_errors=True)
    else:
        shutil.rmtree(subdirectory(run_name, group), ignore_errors=True)

def path(run_name, group, channel=''):
    # Python's idea of a process name is different from the system's idea. Dunno where
    # the difference comes from.
    run_name = resolve(run_name)

    proc = mp.current_process()

    for x in [run_name, group]:
        for c in ['_', os.sep]:
            assert c not in x, f'Can\'t have "{c}" in the file path'

    path = subdirectory(run_name, group, channel) / f'{proc.name}-{proc.pid}'

    path.parent.mkdir(exist_ok=True, parents=True)
    return path

def glob(run_name, group, channel='', pattern='*'):
    paths = subdirectory(run_name, group, channel).glob(pattern)
    return sorted(paths, key=lambda p: p.stat().st_mtime)

def parse(path):
    parts = path.relative_to(ROOT).with_suffix('').parts
    procname, pid = re.match(r'^(.*)-(.*)$', parts[-1]).groups()
    return dotdict.dotdict(
        run_name=parts[0], 
        group=parts[1], 
        channel='/'.join(parts[2:-1]),
        filename=parts[-1],
        procname=procname, 
        pid=pid)

def runs():
    paths = []
    for p in Path(ROOT).iterdir():
        paths.append({
            'path': p, 
            'created': pd.Timestamp(p.stat().st_ctime, unit='s'),
            'run_name': p.parts[-1]})
    return pd.DataFrame(paths).sort_values('created').reset_index(drop=True)

def size(run_name, group):
    run_name = resolve(run_name)
    b = sum(item.stat().st_size for item in subdirectory(run_name, group).glob('**/*.*'))
    return b/1e6
