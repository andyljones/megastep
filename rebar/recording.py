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
from matplotlib import tight_bbox
import numbers
import sys

log = logging.getLogger(__name__)

def adjust_bbox(fig):
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    tight_bbox.adjust_bbox(fig, bbox, fig.canvas.fixed_dpi)

def array(fig):
    adjust_bbox(fig)
    fig.canvas.draw_idle()
    renderer = fig.canvas.get_renderer()
    w, h = int(renderer.width), int(renderer.height)
    # Resolution must even, else libx264 gets upset
    h2, w2 = 2*(h//2), 2*(w//2)
    return (np.frombuffer(renderer.buffer_rgba(), np.uint8)
                        .reshape((h, w, 4))
                        [:h2, :w2, :3]
                        .copy())

class Encoder:

    def __init__(self, fps=20):
        """A context manager for encoding frames of video. Usually you'll want to use :class:`ParallelEncoder` instead.
        
        Typically used as ::

            with Encoder() as encoder:
                # Call it with each frame in turn.
                for frame in frames:
                    encoder(frame)

            # Now write it out.
            with open('test.mp4', 'b') as f:
                f.write(encoder.value)

        In this example, ``frame`` is a (H, W, 1 or 3)-dim numpy array, or a matplotlib figure.
        
        This follows the `PyAV cookbook <http://docs.mikeboers.com/pyav/develop/cookbook/numpy.html#generating-video>`_.
        """
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

def notebook(video, height=640):
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
        """A context manager for encoding frames of video in parallel. Typically used as ::
        
            with ParallelEncoder(f) as encoder:
                for x in xs:
                    encoder(x)
            encoder.notebook()  # to display the video in your notebook
            encoder.save(path)  # to save the video
        
        In this example, ``f`` is a function that takes some arguments and returns a (H, W, 1 or 3)-dim numpy array, 
        or a matplotlib figure. Whatever you call ``encoder`` with will be forwarded to ``f`` in a separate process,
        and the resulting array will be brought back to this process for encoding.

        This aligns with the common scenario where generating each frame with matplotlib is much slower than actually
        getting the arguments needed to do the generation, or doing the encoding itself. 

        :param fps: The framerate. Defaults to 20.
        :type fps: int
        :param N: The number of processes to use. Can be an integer or a float indicating the fraction of CPUs to
            use. Defaults to using 1/2 the CPUs.
        :type N: int, float
        """
        cpus = multiprocessing.cpu_count()
        if N is None:
            N = cpus//2
        elif isinstance(N, numbers.Integral):
            N = N
        elif isinstance(N, numbers.Real):
            N = int(cpus*N)
        else:
            raise ValueError(f'Number of processes must be an integer, a float, or None. Got a "{type(N)}"')

        self._encoder = Encoder(fps)
        self._f = f
        self._queuelen = N

        # Only Python >=3.7 has initializers for the process pool. 
        if sys.version_info[1] < 7:
            self._pool = parallel(_array, progress=False, N=N)
        else:
            self._pool = parallel(_array, progress=False, N=N, initializer=_init)

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