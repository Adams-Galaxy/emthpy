"""Contains data structures for Engineering Mathimatics"""

import numpy as np
import math
import _emth_exceptions as ex


def sign(x): return int(x / abs(x)) if x != 0 else 1


class Vector(np.ndarray):
    """Class for working with vectors, and 
    utiliasing vector operations"""

    def __new__(cls, *args, dtype=float, **kwargs):
        if len(args) > 1:
            return np.asarray(args, **kwargs, dtype=dtype).view(cls)
        elif not isinstance(args[0], (list, tuple, np.ndarray)):
            if isinstance(args[0], (int, float)):
                raise ValueError("vector must be a minimum of 2-space")
            raise TypeError(f"expected array-like object not {type(args[0]).__name__}")
        elif len(args[0]) < 2:
            raise ValueError("vector must be a minimum of 2-space")
        return np.asarray(*args, **kwargs, dtype=dtype).view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        
    @property
    def magnitude(self):
        return math.sqrt(np.sum(self))
    @property
    def r_space(self):
        return self.shape[0]

    @property
    def x(self):
        return self[0]
    @x.setter
    def x(self, value):
        self[0] = value
    
    @property
    def y(self):
        return self[1]
    @y.setter
    def y(self, value):
        self[1] = value
    
    @property
    def z(self):
        if self.r_space < 3:
            raise RSpaceError(f"vector in {self.r_space}-space has no 'z' component")
        return self[2]
    @z.setter
    def z(self, value):
        if self.r_space < 3:
            raise RSpaceError(f"vector in {self.r_space}-space has no 'z' component")
        self[0] = value
    
    @property
    def w(self):
        if self.r_space < 4:
            raise RSpaceError(f"vector in {self.r_space}-space has no 'w' component")
        return self[3]
    @w.setter
    def w(self, value):
        if self.r_space < 4:
            raise RSpaceError(f"vector in {self.r_space}-space has no 'w' component")
        self[3] = value

    def normalise(self):
        Vector.vec_normalise(self)
    def normalised(self):
        result = Vector(self)
        result.normalise()
        return result

    @classmethod
    def zero(cls, r_space, **kwargs):
        return cls(np.zeros((r_space)), **kwargs)
    @classmethod
    def one(cls, r_space, **kwargs):
        return cls(np.ones((r_space)), **kwargs)
    @classmethod
    def full(cls, r_space, value, **kwargs):
        return cls(np.full((r_space), value), **kwargs)

    @classmethod
    def vec_cross(cls, a, b):
        """Return the resultant of 'a' crossed with 'b'. 
        This only works for vectors in 3-space."""

        if a.shape != (3,) or b.shape != (3,):
            raise ValueError("Cannot cross non-3-space vectors")

        # Preform the cross product
        result = (a.y*b.z - a.z*b.y, -(a.x*b.z - a.z*b.x), a.x*b.y - a.y*b.x)
        return cls(result)
    @staticmethod
    def vec_normalise(vector):
        vector /= vector.magnitude

class Matrix(np.ndarray):
    """Class for working with matricies, and
    utiliasing matrix operations"""

    def __new__(cls, *args, **kwargs):
        obj = np.asarray(*args, **kwargs).view(cls)
        if len(obj.shape) > 2:
            raise MatrixShapeError(
                f"maxtrix can only be of shape (m, n), not {obj.shape}")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._iter_vector = Vector.zero(2, dtype=int)

    def __iter__(self):
        self._iter_vector = Vector.zero(2, dtype=int)
        return self

    def __next__(self):
        result = (self._iter_vector.y, self._iter_vector.x)
        
        self._iter_vector.x += 1
        if self._iter_vector.x >= self.n:
            self._iter_vector.x = 0
            self._iter_vector.y += 1
        if self._iter_vector.y >= self.m:
            raise StopIteration
        return result

    @property
    def m(self):
        return self.shape[0]
    @property
    def n(self):
        return self.shape[1]

    def is_square(self):
        return Matrix.mat_is_square(self)

    def is_inverse_of(self, other):
        return Matrix.mat_are_inverses(self, other)

    def augmented_with(self, other, clone_data=True):
        if clone_data:
            return AugmentedMatrix(self.copy(), other.copy())
        return AugmentedMatrix(self, other)

    def identity(self):
        return Matrix.mat_identity(self.shape)

    def add_scaled_row_to_another(self, row_a, row_b, scalar):
        self[row_b] += self[row_a] * scalar

    def scale_row(self, row, scalar):
        self[row] *= scalar

    def interchange_rows(self, row_a, row_b):
        temp = self[row_a]
        self[row_a] = self[row_b]
        self[row_b] = temp

    @staticmethod
    def mat_are_inverses(mat_a, mat_b):
        if mat_a.shape != mat_b.shape:
            return False
        a = mat_a * mat_b
        b = mat_a.identity()
        print(a)
        print(b)
        return np.allclose(a, b)

    @staticmethod
    def mat_identity(shape):
        """Get the identity matrix of a matrix of order 'order'."""
        m, n = shape
        if m != n:
            raise ValueError(
                f"Invalid order: {shape}, both 'm', and 'n' must be equal.")

        result = Matrix(np.zeros(shape))
        for i in range(m):
            result[i, i] = 1
        return result

    @staticmethod
    def mat_is_square(matrix):
        return matrix.m == matrix.n

    @staticmethod
    def _get_inversion_steps(shape):
        """Returns a list of points to nullify ordered in sequencial order, for inversion."""

        m, n = shape

        sequence = []
        for j in range(n):
            in_triangle = False
            for i in range(m):
                if (i != j and not in_triangle):  # Wait for piviot
                    continue
                if i == j:
                    in_triangle = True
                    continue

                sequence.append((i, j))
        for j in range(n - 1, -1, -1):
            in_triangle = False
            for i in range(m - 1, -1, -1):
                if (i != j and not in_triangle):  # Wait for piviot
                    continue
                if i == j:
                    in_triangle = True
                    continue
                sequence.append((i, j))
        return sequence

    @staticmethod
    def inverse_mat(matrix, print_steps=False) -> bool:
        """Attempts to inverse 'matrix', returns whether the atempt was successful."""
        if not isinstance(matrix, Matrix):
            raise TypeError(
                f"'matrix' must be of type {Matrix.__name__}, not {type(matrix).__name__}")
        if not Matrix.is_square(matrix):
            raise ValueError(
                f"'matrix' must be square (nxn) to conform for inversion, not {matrix.order}")

        sequence = Matrix._get_inversion_steps(matrix.shape)
        aug_matrix = matrix.augmented_with(matrix.identity())

        if print_steps:
            print(aug_matrix)
        for point in sequence:
            i, j = point
            pivot = aug_matrix[j, j]

            if pivot == 0:
                new_pivot_index = Matrix._check_nonzero_pivot_in_col(
                    aug_matrix, j)
                if new_pivot_index != -1:
                    aug_matrix.interchange_rows(j, new_pivot_index)
                    pivot = aug_matrix[j, j]
                else:
                    return False
            value = aug_matrix[point]
            if value == 0:
                continue

            aug_matrix.add_scaled_row_to_another(j, i,
                                      -sign(value*pivot) * abs(value / pivot))
            if print_steps:
                print(aug_matrix)
        for i in range(aug_matrix.m):
            try:
                aug_matrix[i] /= aug_matrix[i, i]
                if print_steps:
                    print(aug_matrix)
            except ZeroDivisionError:
                return False
        inversed = aug_matrix.b_matrix

        # Check whether the inverse was actualy successful
        if inversed.is_inverse_of(matrix):
            matrix = inversed
            return True
        return False

    @staticmethod
    def _check_nonzero_pivot_in_col(matrix, col, start_index=0):
        for i in range(start_index, matrix.order.m):
            if matrix[i][col] != 0:
                return i
        return -1

class AugmentedMatrix(Matrix):
    """Class for working with augmented matrices."""

    def __new__(cls, mat_a, mat_b, **kwargs):
        obj = super().__new__(cls, np.hstack((mat_a, mat_b)), **kwargs)
        return obj
    
    def __array_finalize__(self, obj):
        return super().__array_finalize__(obj)
    
    def __init__(self, mat_a, mat_b, **kwargs) -> None:
        super().__init__(**kwargs)

        self.mat_a_shape = mat_a.shape
        self.mat_b_shape = mat_b.shape

    @property
    def a_matrix(self):
        """Returns the A component of [A | B], in this augmented matrix."""
        m, n = self.mat_a_shape
        return self[:,:n]

    @property
    def b_matrix(self):
        """Returns the A component of [A | B], in this augmented matrix."""
        m, n = self.mat_b_shape
        return self[:, -n:]

    """def __str__(self) -> str:
        a_mat_strings = str(self.a_matrix).split('\n')[:-1]
        b_mat_strings = str(self.b_matrix).split('\n')[:-1]

        result = ""
        for i in range(len(a_mat_strings)):
            result += a_mat_strings[i][:-1] + \
                " | " + b_mat_strings[i][1:] + '\n'
        return result"""

a = Matrix([
    [1, 2, 3],
    [3, -2, 1],
    [4, 1, 1],
])

print(Matrix.inverse_mat(a, print_steps=True))
