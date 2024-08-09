import numpy as np
from ._emthpy_rationals import Rational
from ._emthpy_base import Evaluable
from ._emthpy_casting import parse_str

class BaseArray(np.ndarray, Evaluable):
    def __new__(cls, *args, **kwargs):
        if 'dtype' in kwargs:
            obj = np.asarray(*args, **kwargs).view(cls)
        else:
            kwargs['dtype'] = object
            obj = np.asarray(*args, **kwargs).view(cls)
            for point in np.ndindex(obj.shape):
                if isinstance(obj[point], str):
                    obj[point] = parse_str(obj[point])
                if not isinstance(obj[point], Rational):
                    obj[point] = Rational(obj[point], 1)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def copy(self):
        return super().copy().view(type(self))

    def evaluate(self, *args, **kwargs):
        for point in np.ndindex(self.shape):
            while isinstance(self[point], Evaluable):
                self[point] = self[point].evaluated(*args, **kwargs)

    def evaluated(self, *args, **kwargs):
        result = self.copy()
        result.evaluate(*args, **kwargs)
        return result

    def reduce(self, *args, **kwargs):
        for point in np.ndindex(self.shape):
            while isinstance(self[point], Evaluable):
                self[point] = self[point].reduced(*args, **kwargs)

    def reduced(self, *args, **kwargs):
        result = self.copy()
        result.reduce(*args, **kwargs)
        return result
    
    def __call__(self, *args, **kwargs):
        return self.evaluated(*args, **kwargs)
