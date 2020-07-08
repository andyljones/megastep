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
    @wraps(f)
    def g(x, *args, **kwargs):
        if isinstance(x, dict):
            return type(x)([(k, g(v, *args, **kwargs)) for k, v in x.items()])
        if isinstance(f, str):
            return getattr(x, f)(*args, **kwargs)
        return f(x, *args, **kwargs)
    return g

def starmapping(f):
    @wraps(f)
    def g(x, *args):
        if isinstance(x, dict):
            return type(x)([(k, g(x[k], *(a[k] for a in args))) for k in x])
        if isinstance(f, str):
            return getattr(x, f)(*args)
        else:
            return f(x, *args)
    return g

def leaves(t):
    if isinstance(t, dict):
        return [l for v in t.values() for l in leaves(v)]
    return [t]

class dotdict(OrderedDict):
    """A dotdict is a dictionary with dot (attribute) access to its elements. 

    You can access elements with either `[key]` or `.key` notation, but always assign new values with `d[key] = v` . 
    This convention is entirely my taste, but it aligns well with the usual use-case of assigning values rarely and 
    reading them regularly.

    There are many dotdict implementations on the internet. This one is special because when you nest dotdicts 
    inside themselves, they're printed prettily:

    >>> dotdict(a=dotdict(b=1, c=2), d=3)
    dotdict:
    a    dotdict:
        b    1
        c    2
    d    3

    It's especially pretty when some of your elements are collections, possibly with shapes and dtypes:

    >>> dotdict(a=np.array([1, 2]), b=torch.as_tensor([3., 4., 5.])) 
    TODO: This

    On top of the pretty-printing, this dotdict has a couple of methods for making `method-chaining
    <https://tomaugspurger.github.io/method-chaining.html>`_ easier. Method chaining is nice because the computation
    flows left-to-right, top-to-bottom. Setting up an example `d`,

    >>> d = dotdict(a=dotdict(b=1, c=2), d=3)

    then you can act on the entire datastructure

    >>> d.pipe(list)                # Act on the entire datastructure
    ['a', 'd']

    or act on the leaves

    >>> d.map(float)                # Act on the leaves
    TODO: This

    or combine it with another dotdict

    >>> d.starmap(int.__add__, d)  
    TODO: This

    or, together

    >>> (d
            .map(float)
            .starmap(float.__add__, d)
            .pipe(list))
    TODO: This

    You generally use dotdict in places that _really_ you should use a `namedtuple`, except that forcing explicit types on
    things would reduce your iteration velocity. Using a dictionary instead lets you keep things flexible. The
    principal costs are that you lose type-safety, and your keys might clash with method names.
    
    TODO: Write a better tradeoff section.
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
        return type(self)(super().copy()) 
    
    def pipe(self, f, *args, **kwargs):
        return f(self, *args, **kwargs)

    def starmap(self, f, *args):
        return starmapping(f)(self, *args)

    def map(self, f, *args, **kwargs):
        return mapping(f)(self, *args, **kwargs)