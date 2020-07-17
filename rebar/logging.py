import copy
import logging
import time
from pathlib import Path
from collections import defaultdict, deque
import logging.handlers
import ipywidgets as widgets
from contextlib import contextmanager
import psutil
from . import widgets, paths
from .contextlib import maybeasynccontextmanager
import sys
import traceback
import _thread
import threading

# for re-export
from logging import getLogger

log = getLogger(__name__)

#TODO: This shouldn't be at the top level
logging.basicConfig(
            stream=sys.stdout, 
            level=logging.INFO, 
            format='%(asctime)s %(levelname)s %(name)s: %(message)s', 
            datefmt=r'%Y-%m-%d %H:%M:%S')
logging.getLogger('parso').setLevel('WARN')  # Jupyter's autocomplete spams the output if this isn't set
log.info('Set log params')

def in_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

class StdoutRenderer:

    def __init__(self):
        super().__init__()

    def emit(self, path, line):
        source = '{procname}/#{pid}'.format(**paths.parse(path))
        print(f'{source}: {line}')

    def close(self):
        pass

class IPythonRenderer:

    def __init__(self, compositor=None):
        super().__init__()
        self._out = (compositor or widgets.Compositor()).output()
        self._next = time.time()
        self._lasts = {}
        self._buffers = defaultdict(lambda: deque(['']*self._out.lines, maxlen=self._out.lines))

    def _format_block(self, name):
        n_lines = max(self._out.lines//(len(self._buffers) + 2), 1)
        lines = '\n'.join(list(self._buffers[name])[-n_lines:])
        return f'{name}:\n{lines}'

    def _display(self, force=False):
        content = '\n\n'.join([self._format_block(n) for n in self._buffers])
        self._out.refresh(content)

        for name, last in list(self._lasts.items()):
            if time.time() - last > 120:
                del self._buffers[name]
                del self._lasts[name]

    def emit(self, path, line):
        source = '{procname}/#{pid}'.format(**paths.parse(path))
        self._buffers[source].append(line)
        self._lasts[source] = time.time()
        self._display()
    
    def close(self):
        self._display(force=True)
        # Want to leave the outputs open so you can see the final messages
        # self._out.close()
        super().close()

@contextmanager
def handlers(*new_handlers):
    logger = logging.getLogger()
    old_handlers = [*logger.handlers]
    try:
        logger.handlers = new_handlers
        yield 
    finally:
        for h in new_handlers:
            try:
                h.acquire()
                h.flush()
                h.close()
            except (OSError, ValueError):
                pass
            finally:
                h.release()

        logger.handlers = old_handlers

@maybeasynccontextmanager
def to_dir(run_name):
    path = paths.path(run_name, 'logs').with_suffix('.txt')
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(name)s: %(message)s', 
        datefmt=r'%H:%M:%S'))

    with handlers(handler):
        try:
            yield
        except:
            log.info(f'Trace:\n{traceback.format_exc()}')
            raise

class Reader:

    def __init__(self, run_name):
        self._dir = paths.subdirectory(run_name, 'logs')
        self._files = {}

    def read(self):
        for path in self._dir.glob('*.txt'):
            if path not in self._files:
                self._files[path] = path.open('r')
        
        for path, f in self._files.items():
            for line in f.readlines():
                yield path, line.rstrip('\n')

def __from_dir(canceller, renderer, reader):
    while True:
        for path, line in reader.read():
            renderer.emit(path, line)

        if canceller.is_set():
            break

        time.sleep(.01)

def _from_dir(canceller, renderer, reader):
    try:
        __from_dir(canceller, renderer, reader)
    except KeyboardInterrupt:
        log.info('Interrupting main')
        _thread.interrupt_main()
        __from_dir(canceller, renderer, reader)

@contextmanager
def from_dir(run_name, compositor=None):
    if in_ipython():
        renderer = IPythonRenderer(compositor)
    else:
        renderer = StdoutRenderer()

    with to_dir(run_name):
        try:
            reader = Reader(run_name)
            canceller = threading.Event()
            thread = threading.Thread(target=_from_dir, args=(canceller, renderer, reader))
            thread.start()
            yield
        finally:
            log.info('Cancelling log forwarding thread')
            time.sleep(.25)
            canceller.set()
            thread.join(1)
            if thread.is_alive():
                log.error('Logging thread won\'t die')
            else:
                log.info('Log forwarding thread cancelled')

@contextmanager
def via_dir(run_name, compositor=None):
    with to_dir(run_name), from_dir(run_name, compositor):
        yield

### TESTS

def test_in_process():
    paths.clear('test', 'logs')

    with from_dir('test'):
        for _ in range(10):
            log.info('hello')
            time.sleep(.1)

def _test_multiprocess(run_name):
    with to_file(run_name):
        for i in range(10):
            log.info(str(i))
            time.sleep(.5)

def test_multiprocess():
    paths.clear('test', 'logs')

    import multiprocessing as mp
    with from_dir('test'):
        ps = []
        for _ in range(3):
            p = mp.Process(target=_test_multiprocess, args=('test',))
            p.start()
            ps.append(p)

        while any(p.is_alive() for p in ps):
            time.sleep(.5)

def _test_error(run_name):
    with to_file(run_name):
        log.info('Alive')
        time.sleep(2)
        raise ValueError('Last gasp')

def test_error():
    paths.clear('test', 'logs')

    import multiprocessing as mp
    with from_dir('test'):
        ps = []
        for _ in range(1):
            p = mp.Process(target=_test_error, args=('test',))
            p.start()
            ps.append(p)

        while any(p.is_alive() for p in ps):
            time.sleep(.5)
