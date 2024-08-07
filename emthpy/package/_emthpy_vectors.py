from math import sqrt
import numpy as np
from . import _emthpy_exceptions as ex
from ._emthpy_base_array import BaseArray


class Vector(BaseArray):
    """
    Class for working with vectors and utilizing vector operations.

    This class extends numpy's ndarray to provide additional vector-specific
    operations and properties.

    Examples:
        >>> p = Vector(-1, 2, -3)
        >>> str(p)
        (-1, 2, -3)
        >>> p.x
        -1
        >>> v = Vector([1, 2, 3])
        >>> v.magnitude
        3.7416573867739413
        >>> v.normalised()
        Vector(0.2672612419124244, 0.5345224838248488, 0.8017837257372732)

    Notes:
        - This class assumes vectors are at least 2-dimensional.
        - The class provides properties for accessing and setting x, y, z, and w components.
    """

    def __new__(cls, *args, **kwargs):
        """
        Create a new Vector instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Vector: A new Vector instance.

        Raises:
            ValueError: If the vector is less than 2-dimensional.
            TypeError: If the input is not an array-like object.
        """
        obj = super().__new__(cls, *args, **kwargs)
        
        if obj.ndim == 0:
            raise ValueError("vector must be a minimum of 2-space")
        return obj.view(cls)

    @property
    def magnitude(self):
        """
        Return the magnitude of the vector.

        Returns:
            float: The magnitude of the vector.
        """
        return sqrt(np.sum(self**2))

    @property
    def r_space(self):
        """
        Return the dimension of the vector.

        Returns:
            int: The dimension of the vector.
        """
        return self.shape[0]

    @property
    def x(self):
        """
        Return the x component of the vector.

        Returns:
            float: The x component of the vector.
        """
        return self[0]

    @x.setter
    def x(self, value):
        """
        Set the x component of the vector.

        Args:
            value (float): The value to set the x component to.
        """
        self[0] = value

    @property
    def y(self):
        """
        Return the y component of the vector.

        Returns:
            float: The y component of the vector.
        """
        return self[1]

    @y.setter
    def y(self, value):
        """
        Set the y component of the vector.

        Args:
            value (float): The value to set the y component to.
        """
        self[1] = value

    @property
    def z(self):
        """
        Return the z component of the vector.

        Returns:
            float: The z component of the vector.

        Raises:
            RSpaceError: If the vector is not in 3-space.
        """
        if self.r_space < 3:
            raise ex.RSpaceError(
                f"vector in {self.r_space}-space has no 'z' component")
        return self[2]

    @z.setter
    def z(self, value):
        """
        Set the z component of the vector.

        Args:
            value (float): The value to set the z component to.

        Raises:
            RSpaceError: If the vector is not in 3-space.
        """
        if self.r_space < 3:
            raise ex.RSpaceError(
                f"vector in {self.r_space}-space has no 'z' component")
        self[2] = value

    @property
    def w(self):
        """
        Return the w component of the vector.

        Returns:
            float: The w component of the vector.

        Raises:
            RSpaceError: If the vector is not in 4-space.
        """
        if self.r_space < 4:
            raise ex.RSpaceError(
                f"vector in {self.r_space}-space has no 'w' component")
        return self[3]

    @w.setter
    def w(self, value):
        """
        Set the w component of the vector.

        Args:
            value (float): The value to set the w component to.

        Raises:
            RSpaceError: If the vector is not in 4-space.
        """
        if self.r_space < 4:
            raise ex.RSpaceError(
                f"vector in {self.r_space}-space has no 'w' component")
        self[3] = value

    def normalize(self):
        """
        Normalize the vector.

        This method modifies the vector in place.
        """
        Vector.vec_normalize(self)

    def normalized(self):
        """
        Return a normalized copy of the vector.

        Returns:
            Vector: A normalized copy of the vector.
        """
        result = self / self.magnitude
        return result

    def projected_onto(self, other):
        """
        Return the projection of the vector onto another vector.

        Args:
            other (Vector): The vector to project onto.

        Returns:
            Vector: The projection of the vector onto another vector.
        """
        return Vector.vec_projection(self, other)

    @classmethod
    def zero(cls, r_space=3, **kwargs):
        """
        Return a zero vector of the specified dimension.

        Args:
            r_space (int): The dimension of the vector.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Vector: A zero vector of the specified dimension.
        """
        return cls(np.zeros((r_space)), **kwargs)

    @classmethod
    def one(cls, r_space=3, **kwargs):
        """
        Return a vector with all components set to 1.

        Args:
            r_space (int): The dimension of the vector.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Vector: A vector with all components set to 1.
        """
        return cls(np.ones((r_space)), **kwargs)

    @classmethod
    def full(cls, value, r_space=3, **kwargs):
        """
        Return a vector with all components set to the specified value.

        Args:
            r_space (int): The dimension of the vector.
            value (float): The value to set all components to.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Vector: A vector with all components set to the specified value.
        """
        return cls(np.full((r_space), value), **kwargs)

    @staticmethod
    def vec_cross(a, b, **kwargs):
        """
        Return the cross product of two vectors.

        Args:
            a (Vector): The first vector.
            b (Vector): The second vector.

        Returns:
            Vector: The cross product of the two vectors.

        Raises:
            ValueError: If either vector is not in 3-space.
        """
        return np.cross(a, b, **kwargs).view(Vector)

    @staticmethod
    def vec_projection(a, b, **kwargs):
        """
        Return the projection of vector a onto vector b.

        Args:
            a (Vector): The vector to project.
            b (Vector): The vector to project onto.

        Returns:
            Vector: The projection of vector a onto vector b.
        """
        return (a @ b) / (b @ b) * b

    @staticmethod
    def vec_normalize(vector):
        """
        Normalize the vector.

        Args:
            vector (Vector): The vector to normalize.
        """
        vector /= vector.magnitude

    def __repr__(self):
        """
        Return the string representation of the vector.

        Returns:
            str: The string representation of the vector.
        """
        return f"Vector{tuple(self)}"

    def __str__(self):
        """
        Return the string form of the vector.

        Returns:
            str: The string form of the vector.
        """
        return f"({', '.join(str(i) for i in self)})"
    
    def __matmul__(self, other):
        return np.dot(self, other)
    
    def __mul__(self, other):
        if isinstance(other, Vector):
            return self @ other
        return super().__mul__(other)


# i = Vector(1, 0, 0)
# j = Vector(0, 1, 0)
# k = Vector(0, 0, 1)
# 
# zero = Vector.zero()
# one = Vector.one()