import numpy as np
import _emthpy_exceptions as ex

class Vector(np.ndarray):
    """Class for working with vectors and utilizing vector operations"""

    def __new__(cls, *args, dtype=float, **kwargs):
        if len(args) > 1:
            return np.asarray(args, **kwargs, dtype=dtype).view(cls)
        elif not isinstance(args[0], (list, tuple, np.ndarray)):
            if isinstance(args[0], (int, float)):
                raise ValueError("vector must be a minimum of 2-space")
            raise TypeError(
                f"expected array-like object not {type(args[0]).__name__}")
        elif len(args[0]) < 2:
            raise ValueError("vector must be a minimum of 2-space")
        return np.asarray(*args, **kwargs, dtype=dtype).view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def magnitude(self):
        """Return the magnitude of the vector"""
        return np.sqrt(np.sum(self))

    @property
    def r_space(self):
        """Return the dimension of the vector"""
        return self.shape[0]

    @property
    def x(self):
        """Return the x component of the vector"""
        return self[0]

    @x.setter
    def x(self, value):
        """Set the x component of the vector"""
        self[0] = value

    @property
    def y(self):
        """Return the y component of the vector"""
        return self[1]

    @y.setter
    def y(self, value):
        """Set the y component of the vector"""
        self[1] = value

    @property
    def z(self):
        """Return the z component of the vector"""
        if self.r_space < 3:
            raise ex.RSpaceError(
                f"vector in {self.r_space}-space has no 'z' component")
        return self[2]

    @z.setter
    def z(self, value):
        """Set the z component of the vector"""
        if self.r_space < 3:
            raise ex.RSpaceError(
                f"vector in {self.r_space}-space has no 'z' component")
        self[2] = value

    @property
    def w(self):
        """Return the w component of the vector"""
        if self.r_space < 4:
            raise ex.RSpaceError(
                f"vector in {self.r_space}-space has no 'w' component")
        return self[3]

    @w.setter
    def w(self, value):
        """Set the w component of the vector"""
        if self.r_space < 4:
            raise ex.RSpaceError(
                f"vector in {self.r_space}-space has no 'w' component")
        self[3] = value

    def normalise(self):
        """Normalize the vector"""
        Vector.vec_normalise(self)

    def normalised(self):
        """Return a normalized copy of the vector"""
        result = Vector(self)
        result.normalise()
        return result

    @classmethod
    def zero(cls, r_space, **kwargs):
        """Return a zero vector of the specified dimension"""
        return cls(np.zeros((r_space)), **kwargs)

    @classmethod
    def one(cls, r_space, **kwargs):
        """Return a vector with all components set to 1"""
        return cls(np.ones((r_space)), **kwargs)

    @classmethod
    def full(cls, r_space, value, **kwargs):
        """Return a vector with all components set to the specified value"""
        return cls(np.full((r_space), value), **kwargs)

    @classmethod
    def vec_cross(cls, a, b):
        """Return the cross product of two vectors"""
        if a.shape != (3,) or b.shape != (3,):
            raise ValueError("Cannot cross non-3-space vectors")

        # Perform the cross product
        result = (a.y * b.z - a.z * b.y, -
                  (a.x * b.z - a.z * b.x), a.x * b.y - a.y * b.x)
        return cls(result)

    @staticmethod
    def vec_normalise(vector):
        """Normalize the vector"""
        vector /= vector.magnitude

    def __repr__(self):
        return f"Vector{tuple(self)}"
    
    def __str__(self):
        return f"{tuple(self)}"
