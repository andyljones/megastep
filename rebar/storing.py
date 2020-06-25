import pandas as pd
import pickle
from . import paths
import time

def store_latest(run_name, objs, throttle=0):
    path = paths.path(run_name, 'storing').with_suffix('.pkl')
    if path.exists():
        if (time.time() - path.lstat().st_mtime) < throttle:
            return False

    state_dicts = {k: v.state_dict() for k, v in objs.items()}
    bs = pickle.dumps(state_dicts)
    path.with_suffix('.tmp').write_bytes(bs)
    path.with_suffix('.tmp').rename(path)

    return True

def runs():
    return paths.runs()

def stored(run_name=-1):
    ps = paths.subdirectory(run_name, 'storing').glob('*.pkl')
    infos = []
    for p in ps:
        infos.append({
            **paths.parse(p),
            'path': p})

    return pd.DataFrame(infos)

def load(run_name=-1, procname='MainProcess'):
    path = stored(run_name).loc[lambda df: df.procname == procname].iloc[-1].path
    return pickle.loads(path.read_bytes())