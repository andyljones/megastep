===
FAQ
===

.. _install:

How do I install megastep?
--------------------------
If you're working on an Ubuntu machine with CUDA already set up, it should be as simple as ::

    pip install megastep

or if you want the :ref:`demo dependencies <subpackages>` too

    pip install megastep[cubicasa,rebar]

There are some fairly hefty dependencies so it might take a while, but it should get there fine. If it
doesn't, my total guess is that it's the Pytorch install that will be the problem, in which case I refer you to
`their install page <https://pytorch.org/get-started/locally/>`_. 

**If you haven't got CUDA, megastep will not work**. There are some parts of megastep - like the cubicasa package - 
that you may still find useful, but in that case I recommend just copy-pasting the code you want from Github.

If you're on a different OS, then it's possible megastep will work, but I can't provide you any support. You're welcome
to ask for help on the GitHub issues page, but you'll be relying on the community to come up with an answer.

.. _inheritance:

Why doesn't megastep use inheritance?
-------------------------------------
TODO: Figure out a better way to phrase this

One way to think about inheritance in software development is that's about offering a secondary interface to a class.

The primary interface to a class is its public methods. When you write a class for some other part of your program to 
use, this is usually what you have in mind.

When you encourage people to inherit from your classes though, you're effectively declaring a secondary interface,
saying 'here are some useful ways to exploit its private state'.

The thing is, Python's ideas of public and private amount to gentle suggestions. This leads to a tertiary interface
to every class, which is where you totally ignore what the designer of the class intended you to do and rely instead
entirely on how the class *actually works*. You freely read and write its private state, monkey patch its methods and
generally let slip the dogs of terrible software development.

Researchers are very fond of this tertiary wild-west interface. The reason researchers are fond of it is either
because they're terrible developers (the popular answer), or because research code is a very unusual kind of code.
It's `written many times and read once (if ever) <https://devblogs.microsoft.com/oldnewthing/20070406-00/?p=27343>`_,
it's typically written by one person in a short period of time and it's typically only a few thousand lines of code
that are understood inside and out. Because of this, researchers can happily trade off a lot of otherwise-good
development practices in favour of iteration velocity - the ability to adapt your codebase to a new idea quickly and
easily.

Since **megastep** is explicitly intended to be a foundation for research, it's designed with the third interface in mind.
There are few private methods, and any state that is likely interesting to a user is there for the taking.

.. _openai-gym:

Why don't you use the OpenAI Gym interface?
---------------------------------------------


.. _cubicasa-license:

What's with the cubicasa license?
---------------------------------