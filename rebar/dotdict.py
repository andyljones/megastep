from collections import OrderedDict
from functools import wraps

SCREEN_WIDTH = 119
SCREEN_HEIGHT = 200

def treestr(t):
    """Stringifies a tree structure. These turn up all over the place in my code, so it's worth factoring out"""
    key_length = max(map(len, map(str, t.keys()))) if t.keys() else 0
    max_spaces = 4 + key_length
    val_length = SCREEN_WIDTH - max_spaces
    
    d = {}
    for k, v in t.items():
        if isinstance(v, dotdict):
            d[k] = str(v)
        elif isinstance(v, (list, set, dict)):
            d[k] = f'{type(v).__name__}({len(v)},)'
        elif hasattr(v, 'shape') and hasattr(v, 'dtype'):                    
            d[k] = f'{type(v).__name__}({tuple(v.shape)}, {v.dtype})'
        elif hasattr(v, 'shape'):
            d[k] = f'{type(v).__name__}({tuple(v.shape)})'
        else:
            lines = str(v).splitlines()
            if (len(lines) > 1) or (len(lines[0]) > val_length):
                d[k] = lines[0][:val_length] + ' ...'
            else:
                d[k] = lines[0]

    s = [f'{type(t).__name__}:']
    for k, v in d.items():
        lines = v.splitlines() or ['']
        s.append(str(k) + ' '*(max_spaces - len(str(k))) + lines[0])
        for l in lines[1:]:
            s.append(' '*max_spaces + l)
        if len(s) >= SCREEN_HEIGHT-1:
            s.append('...')
            break

    return '\n'.join(s)

def mapping(f):
    """Wraps ``f`` so that when called on a dotdict, ``f`` instead gets called on the dotdict's values
    and a dotdict of the results is returned. Extra ``*args`` and ``**kwargs`` passed to the wrapper are
    passed as extra arguments to ``f`` .

    >>> d = dotdict(a=1, b=2)
    >>> m = mapping(int.__add__)
    >>> m(d, 10)
    dotdict:
    a    11
    b    12
    
    Works equally well on trees of dotdicts, where ``f`` will be applied to the leaves of the tree.

    Can be used as a decorator.

    See :func:`dotdict.map` for an object-oriented version of this function.
    """

    @wraps(f)
    def g(x, *args, **kwargs):
        if isinstance(x, dict):
            return type(x)([(k, g(v, *args, **kwargs)) for k, v in x.items()])
        if isinstance(f, str):
            return getattr(x, f)(*args, **kwargs)
        return f(x, *args, **kwargs)
    return g

def starmapping(f):
    """Wraps ``f`` so that when called on a sequence of dotdicts, ``f`` instead gets called on the dotdict's values
    and a dotdict of the results is returned.

    >>> d = dotdict(a=1, b=2)
    >>> m = starmapping(int.__add__)
    >>> m(d, d)
    dotdict:
    a    2
    b    4
    
    Works equally well on trees of dotdicts, where ``f`` will be applied to the leaves of the trees.

    Can be used as a decorator.

    See :func:`dotdict.starmap` for an object-oriented version of this function.
    """
    @wraps(f)
    def g(x, *args, **kwargs):
        if isinstance(x, dict):
            return type(x)([(k, g(x[k], *(a[k] for a in args))) for k in x])
        if isinstance(f, str):
            return getattr(x, f)(*args)
        else:
            return f(x, *args)
    return g

def leaves(t):
    """Returns the leaves of a tree of dotdicts as a list"""
    if isinstance(t, dict):
        return [l for v in t.values() for l in leaves(v)]
    return [t]

class dotdict(OrderedDict):
    """A dotdict is a dictionary with dot (attribute) access to its elements. 

    You can access elements with either ``d[k]`` or ``d.k`` notation, but always assign new values with `d[key] = v` . 
    This convention is entirely my taste, but it aligns well with the usual use-case of assigning values rarely and 
    reading them regularly.

    As well as the dot access, there are some extra features for making a researcher's life easier.

    **Delegated Attributes**

    If you try to access an attribute that isn't a method of ``dict`` or a key in the dotdict, then the attribute will
    instead be evaluated on every value in the dictionary. This means if you've got a dotdict full of CPU tensors, you
    can send them all to the GPU with a single call:

    >>> cpu_tensors = dotdict(
    >>>     a=torch.tensor([1]), 
    >>>     b=dotdict(
    >>>         c=torch.tensor([2])))
    >>> gpu_tensors = cpu_tensors.cuda()

    Fair warning: be careful not to use keys in your dotdict that clash with the names of methods you're likely to
    use.

    **Method Chaining**

    There are a couple of methods on the dotdict itself for making `method-chaining
    <https://tomaugspurger.github.io/method-chaining.html>`_ easier. Method chaining is nice because the computation
    flows left-to-right, top-to-bottom. Setting up an example `d`,

    >>> d = dotdict(
    >>>     a=dotdict(
    >>>         b=1, 
    >>>         c=2), 
    >>>     d=3)

    then you can act on the entire datastructure

    >>> d.pipe(list)
    ['a', 'd']

    or act on the leaves

    >>> d.map(float)
    dotdict:
    a    dotdict:
        b    1.0
        c    2.0
    d    3.0

    or combine it with another dotdict

    >>> d.starmap(int.__add__, d)  
    dotdict:
    a    dotdict:
        b    2
        c    4
    d    6

    or, together

    >>> (d
            .map(float)
            .starmap(float.__add__, d)
            .pipe(list))
    ['a', 'd']
    
    **Pretty-printing**

    As you've likely noticed, when you nest dotdicts inside themselves then they're printed prettily:

    >>> dotdict(a=dotdict(b=1, c=2), d=3)
    dotdict:
    a    dotdict:
        b    1
        c    2
    d    3

    It's especially pretty when some of your elements are collections, possibly with shapes and dtypes:

    >>> dotdict(a=np.array([1, 2]), b=torch.as_tensor([[3., 4., 5.]])) 
    dotdict:
    a    ndarray((2,), int64)
    b    Tensor((1, 3), torch.float32)

    **Use cases**

    You generally use dotdict in places that *really* you should use a `namedtuple`, except that forcing explicit types on
    things would make it harder to change things as you go. Using a dictionary instead lets you keep things flexible. The
    principal costs are that you lose type-safety, and your keys might clash with method names.
    """
    
    def __dir__(self):
        return sorted(set(super().__dir__() + list(self.keys())))

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            try:
                return type(self)([(k, getattr(v, key)) for k, v in self.items()])
            except AttributeError:
                raise AttributeError(f"There is no member called '{key}' and one of the leaves has no attribute '{key}'") from None

    def __call__(self, *args, **kwargs):
        return type(self)([(k, v(*args, **kwargs)) for k, v in self.items()])

    def __str__(self):
        return treestr(self)
    
    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
    
    def copy(self):
        """Shallow-copy the dotdict"""
        return type(self)(super().copy()) 
    
    def pipe(self, f, *args, **kwargs):
        """Returns ``f(self, *args, **kwargs)`` . 

        >>> d = dotdict(a=1, b=2)
        >>> d.pipe(list)
        ['a', 'b']

        Useful for method-chaining."""
        return f(self, *args, **kwargs)

    def map(self, f, *args, **kwargs):
        """Applies ``f`` to the values of the dotdict, returning a matching dotdict of the results.
        ``*args`` and  ``**kwargs`` are passed as extra arguments to each call.

        >>> d = dotdict(a=1, b=2)
        >>> d.map(int.__add__, 10)
        dotdict:
        a    11
        b    12

        Useful for method-chaining. Works equally well on trees of dotdicts.
        
        See :func:`mapping` for a functional version of this method."""
        return mapping(f)(self, *args, **kwargs)

    def starmap(self, f, *args, **kwargs):
        """Applies ``f`` to the values of the dotdicts one key at a time, returning a matching dotdict of the results.

        >>> d = dotdict(a=1, b=2)
        >>> d.starmap(int.__add__, d)
        dotdict:
        a    2
        b    4

        Useful for method-chaining. Works equally well on trees of dotdicts.
        
        See :func:`starmapping` for a functional version of this method."""
        return starmapping(f)(self, *args, **kwargs)
