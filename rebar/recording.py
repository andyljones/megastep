import av
from io import BytesIO
import numpy as np
import base64
from IPython.display import display, HTML
from pathlib import Path
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from .parallel import parallel
import logging
import time
import multiprocessing
from pathlib import Path

log = logging.getLogger(__name__)

def array(fig):
    fig.canvas.draw_idle()
    renderer = fig.canvas.get_renderer()
    w, h = int(renderer.width), int(renderer.height)
    return (np.frombuffer(renderer.buffer_rgba(), np.uint8)
                        .reshape((h, w, 4))
                        [:, :, :3]
                        .copy())

class Encoder:

    def __init__(self, fps):
        """This follows the [PyAV cookbook](http://docs.mikeboers.com/pyav/develop/cookbook/numpy.html#generating-video)"""
        self._fps = fps
        self._initialized = False

    def _initialize(self, arr):
        self._content = BytesIO()
        self._container = av.open(self._content, 'w', 'mp4')

        self._stream = self._container.add_stream('h264', rate=self._fps)
        self._stream.pix_fmt = 'yuv420p'
        self._stream.height = arr.shape[0]
        self._stream.width = arr.shape[1]

        pixelformats = {1: 'gray', 3: 'rgb24'}
        self._pixelformat = pixelformats[arr.shape[2]]

        self.height, self.width = arr.shape[:2]
        self.mimetype = 'mp4'

        self._initialized = True
    
    def __enter__(self):
        return self

    def __call__(self, arr):
        if isinstance(arr, plt.Figure):
            fig = arr
            arr = array(fig)
            fig.gcf()

        if not self._initialized:
            self._initialize(arr)

        # Float arrs are assumed to have a domain of [0, 1], for backward-compatability with OpenCV.
        if np.issubdtype(arr.dtype, np.floating):
            arr = (255*arr)
        if not np.issubdtype(arr.dtype, np.uint8):
            arr = arr.astype(np.uint8).clip(0, 255)

        frame = av.VideoFrame.from_ndarray(arr, format=self._pixelformat)
        self._container.mux(self._stream.encode(frame))

    def __exit__(self, type, value, traceback):
        # Flushing the stream here causes a deprecation warning in ffmpeg
        # https://ffmpeg.zeranoe.com/forum/viewtopic.php?t=3678
        # It's old and benign and possibly only apparent in homebrew-installed ffmpeg?
        if not type:
            if hasattr(self, '_container'):
                self._container.mux(self._stream.encode())
                self._container.close()
                self.value = self._content.getvalue()
        
def html_tag(video, height=None, **kwargs):
    video = video.value if isinstance(video, Encoder) else video
    style = f'style="height: {height}px"' if height else ''
    b64 = base64.b64encode(video).decode('utf-8')
    return f"""
<video controls autoplay loop {style}>
    <source type="video/mp4" src="data:video/mp4;base64,{b64}">
    Your browser does not support the video tag.
</video>"""

def notebook(video, height=960):
    return display(HTML(html_tag(video, height)))

def _init():
    # Suppress keyboard interrupt of workers, since exiting the context 
    # manager in the parent will shut them down.
    import signal
    signal.signal(signal.SIGINT, lambda h, f: None)

def _array(f, *args, **kwargs):
    result = f(*args, **kwargs)
    if isinstance(result, plt.Figure):
        arr = array(result)
        plt.close(result)
        return arr
    else:
        return result

class ParallelEncoder:

    def __init__(self, f, fps=20, N=None):
        N = multiprocessing.cpu_count()//4 if N is None else N
        self._encoder = Encoder(fps)
        self._f = f
        self._pool = parallel(_array, progress=False, N=N, initializer=_init)
        self._queuelen = N

    def __enter__(self):
        self._futures = {}
        self._submitted = 0
        self._contiguous = 0

        self._encoder.__enter__()
        self._submit = self._pool.__enter__()
        return self

    def _process_done(self):
        while True:
            if (self._contiguous in self._futures) and self._futures[self._contiguous].done():
                result = self._futures[self._contiguous].result()
                self._encoder(result)
                del self._futures[self._contiguous]
                self._contiguous += 1
            else:
                break

    def _wait(self):
        while self._futures:
            self._process_done()
            time.sleep(.01)

    def __exit__(self, t, v, tb):
        self._wait()
        self._encoder.__exit__(t, v, tb)
        self._pool.__exit__(t, v, tb)

    def __call__(self, *args, **kwargs):
        while len(self._futures) > self._queuelen:
            self._process_done()

        self._futures[self._submitted] = self._submit(self._f, *args, **kwargs)
        self._submitted += 1
        self._process_done()

    def result(self):
        self._wait()
        return self._encoder.value

    def notebook(self):
        return notebook(self.result())

    def save(self, path):
        Path(path).write_bytes(self.result())



def encode(f, *indexable, fps=20, N=0, n_frames=None, **kwargs):
    """To use this with N > 0, you need to return an array and - if it's a new figure each time - 
    close it afterwards"""
    n_frames = len(indexable[0]) if n_frames is None else n_frames
    log.info(f'Encoding begun on {n_frames} frames')
    queuesize = 2*cpu_count() 
    submitted, contiguous = 0, 0
    futures = {}
    with Encoder(fps) as encoder, parallel(f, progress=False, N=N) as p, tqdm(total=n_frames) as pbar:
        while True:
            if (submitted < n_frames) and (len(futures) < queuesize):
                futures[submitted] = p(*[iable[submitted] for iable in indexable], **kwargs)
                submitted += 1
            if (contiguous in futures) and futures[contiguous].done():
                result = futures[contiguous].result()
                if isinstance(result, plt.Figure):
                    fig = result
                    result = array(fig)
                    plt.close(fig)
                encoder(result)
                del futures[contiguous]
                contiguous += 1
                pbar.update(1)
                if (N == 0) and (contiguous % 100 == 0):
                    log.info(f'Finished {contiguous}/{n_frames} frames')
            if contiguous == n_frames:
                log.info('Encoding finished')
                return encoder