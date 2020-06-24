import torch
import pandas as pd
from io import BytesIO
from subprocess import check_output
from . import writing


def memory(name='default'):
    total_mem = torch.cuda.get_device_properties('cuda').total_memory
    writing.max(f'gpu-memory/cache/{name}', torch.cuda.max_memory_cached()/total_mem)
    torch.cuda.reset_max_memory_cached()
    writing.max(f'gpu-memory/alloc/{name}', torch.cuda.max_memory_allocated()/total_mem)
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()

def dataframe():
    """Use `nvidia-smi --help-query-gpu` to get a list of query params"""
    params = {
        'device': 'index', 
        'compute': 'utilization.gpu', 'access': 'utilization.memory', 
        'memused': 'memory.used', 'memtotal': 'memory.total',
        'fan': 'fan.speed', 'power': 'power.draw', 'temp': 'temperature.gpu'}
    command = f"""nvidia-smi --format=csv,nounits,noheader --query-gpu={','.join(params.values())}"""
    df = pd.read_csv(BytesIO(check_output(command, shell=True)), header=None)
    df.columns = list(params.keys())
    df = df.set_index('device')
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def performance():
    df = dataframe()
    fields = ['compute', 'access', 'fan', 'power', 'temp']
    for (device, field), value in df[fields].stack().iteritems():
        writing.mean(f'gpu/{field}/{device}', value)