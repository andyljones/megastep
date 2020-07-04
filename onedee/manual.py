import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from aljpy import arrdict
from ipywidgets import Output, Label, Image, HTML, Play
from IPython.display import display, clear_output
from ipyevents import Event
from PIL import Image as Image_
from io import BytesIO
import torch
import torch.nn.functional as F
import threading

def from_obs(t):
    im = Image_.fromarray(t.mul(255).cpu().numpy().astype(np.uint8))
    bs = BytesIO()
    im.save(bs, 'png')
    return bs.getvalue()

def from_state(env):
    s = env.state()
    fig = env.plot_state(arrdict.numpyify(s)) 
    bs = BytesIO()
    fig.savefig(bs)
    plt.close(fig)
    return bs.getvalue()

def _play(env, action, im, h, fps=20):
    next_frame = time.time()
    count = 0
    while True:
        next_frame = next_frame + 1/fps

        decision = arrdict.arrdict(
            actions=torch.as_tensor(action, device=env.device, dtype=torch.long)[:, None])

        world = env.step(decision)
        h.value = f'#{count}: {action[0]}, {world.reward[0]}, {world.terminal[0]}'
        action[0] = 0
        count += 1

        im.value = from_state(env)

        time.sleep(max(next_frame - time.time(), 0))

def play(env):
    action = np.zeros(env.n_envs, dtype=np.uint8)

    world = env.reset()

    im = Image(value=from_state(env), width=800, height=800)

    h = HTML('Event info')

    def handle(event):
        keys = '123456789'

        if (event['type'] == 'keydown') and (event['key'] in keys):
            action[0] = int(event['key'])

    e = Event(source=im, watched_events=['keydown', 'keyup'])
    e.on_dom_event(handle)

    display(im, h)

    thread = threading.Thread(target=_play, args=(env, action, im, h))
    thread.start()