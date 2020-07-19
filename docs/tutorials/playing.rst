.. _playing:

=====================
Playing With Megastep
=====================
While the :ref:`Minimal Environment <minimal-env>` tutorial is extremely detailed, it's not particularly *fun*. This 
tutorial instead takes an existing interesting environment and tasks you to adapt it in specific ways. In solving
these tasks, you'll learn how megastep is structured while hopefully keeping the problem-solving part of your brain
from losing interest.

Resources
*********
Here are some tools that might help with these tasks.

:ref:`API reference <api>`
    The API reference describes the details of megastep in one place. If you decide you want to alter the `FOV
    <https://en.wikipedia.org/wiki/Field_of_view>`_ of the agents for example, a good way to go about it would be 
    to go to the API reference and Ctrl+F for 'FOV'.

    The API reference also links to the source of the code it's documenting; if you don't find the detail you want
    in the docs themselves, clicking through to the source code will often give you an answer.

:ref:`Concepts <concepts>`
    There are some ideas in megastep - like 'agents' - which turn up in too many places to document them again and 
    again every time they're used. Instead, there's a Concepts page which gives a brief overview of each of these ideas.

:ref:`FAQ <faq>`
    The FAQ tries to preempt some common questions. It remains to be seen how good of a job I've done with it.

:ref:`Tutorials <tutorials>`
    If you're reading this you probably don't want to read the more in-depth tutorials, but they may still be useful
    as something to Ctrl+F through when you're after a specific bit of code.

IPython Help
    You can follow any object in an IPython session `with ? to get the docs for that object, or ?? to get 
    the source code for that object <https://ipython.readthedocs.io/en/stable/interactive/python-ipython-diff.html#accessing-help>`_.

    ?? will also give you the filepath of the code underlying the object, which is useful for the next bit.

Library Breakpoints
    If you're curious how a library is doing a specific thing and just looking at the code isn't helpful, you can 
    use ?? to find the path to the code on your system and open that path in an editor. Then you can `set a breakpoint
    anywhere you want <https://docs.python.org/3/library/pdb.html#pdb.set_trace>`_! This can be done with ``breakpoint()``
    in Python 3.7, or ``import pdb; pdb.set_trace()`` in earlier versions.
    
    If you're using `autoreload <https://ipython.org/ipython-doc/3/config/extensions/autoreload.html>`_ then next time
    you run the code, you'll hit the breakpoint. If you're not using autoreload, you'll either have to use importlib
    to manually reload things, or just restart your IPython session.

    On top of pdb's built-in capabilities, I'd also recommend having a look at `extract <https://andyljones.com/posts/post-mortem-plotting.html>`_.

Demos
    The :github:`demo module <megastep/demo/__init__.py>` has examples of two environments and an example of how to 
    train them.

Setup Tasks
***********

Geometry Tasks
**************

Scenery Tasks
**************

Rendering Tasks
***************

Action Tasks
************

Agent Tasks
***********

Training Tasks
**************

TODO-DOCS Finish the Playing tutorial