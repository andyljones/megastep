import ipywidgets as widgets
from IPython.display import display, clear_output
import threading

WRITE_LOCK = threading.RLock()

class Output:

    def __init__(self, compositor, output, lines):
        self._compositor = compositor
        self._output = output
        self.lines = lines

    def refresh(self, content):
        # This is not thread-safe, but the recommended way to do 
        # thread-safeness - to use append_stdout - causes flickering
        with WRITE_LOCK, self._output:
            clear_output(wait=True)
            print(content)
    
    def close(self):
        self._compositor.remove(self._output)

class Compositor:

    def __init__(self, lines=80):
        self.lines = lines
        self._box = widgets.HBox(
            layout=widgets.Layout(align_items='stretch'))
        display(self._box)

    def output(self):
        output = widgets.Output(
            layout=widgets.Layout(width='100%'))
        self._box.children = (*self._box.children, output)

        return Output(self, output, self.lines)

    def remove(self, child):
        child.close()
        self._box.children = tuple(c for c in self._box.children if c != child)

    def clear(self):
        for child in self._box.children:
            self.remove(child)


def test():
    compositor = Compositor()
    first = compositor.output()
    second = compositor.output()

    first.refresh('left')
    second.refresh('right')