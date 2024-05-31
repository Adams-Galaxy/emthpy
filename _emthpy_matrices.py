import numpy as np
from _emthpy_utils import sign
from _emthpy_vectors import Vector
import _emthpy_exceptions as ex
import _emthpy_utils as utils
import _emthpy_equations as eq

class Matrix(np.ndarray):
    """Class for working with matrices and utilizing matrix operations"""

    def __new__(cls, *args, **kwargs):
        obj = np.asarray(*args, **kwargs).view(cls)
        if len(obj.shape) > 2:
            raise ex.MatrixShapeError(
                f"matrix can only be of shape (m, n), not {obj.shape}")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._iter_vector = Vector.zero(2, dtype=int)

    @classmethod
    def from_existing(cls, matrix):
        """Create a matrix from an existing matrix"""
        return cls(matrix)

    def __iter__(self):
        self._iter_vector = Vector.zero(2, dtype=int)
        return self

    def __next__(self):
        result = (self._iter_vector.y, self._iter_vector.x)

        self._iter_vector.x += 1
        if self._iter_vector.x >= self.n:
            self._iter_vector.x = 0
            self._iter_vector.y += 1
            return result
        if self._iter_vector.y >= self.m:
            raise StopIteration
        return result

    @property
    def m(self):
        """Return the number of rows in the matrix"""
        return self.shape[0]

    @property
    def n(self):
        """Return the number of columns in the matrix"""
        return self.shape[1]

    def is_square(self):
        """Check if the matrix is square"""
        return Matrix.mat_is_square(self)

    def is_inverse_of(self, other):
        """Check if the matrix is the inverse of another matrix"""
        return Matrix.mat_are_inverses(self, other)

    def augmented_with(self, other, clone_data=True):
        """Return an augmented matrix"""
        if clone_data:
            return AugmentedMatrix(self.copy(), other.copy())
        return AugmentedMatrix(self, other)

    def identity(self):
        """Return the identity matrix"""
        return Matrix.mat_identity(self.shape)

    def add_scaled_row_to_another(self, row_a, row_b, scalar):
        """Add a scaled row to another row"""
        self[row_b] += self[row_a] * scalar

    def scale_row(self, row, scalar):
        """Scale a row by a scalar"""
        self[row] *= scalar

    def interchange_rows(self, row_a, row_b):
        """Interchange two rows"""
        temp = self[row_a]
        self[row_a] = self[row_b]
        self[row_b] = temp

    def evaluate(self, *args, **kwargs):
        """Evaluate the matrix"""
        for i, point in enumerate(self):
            if isinstance(self[point], (eq.Equation, utils.Fraction)):
                self[point] = self[point].evaluate(*args, **kwargs)

    def evaluated(self, *args, **kwargs):
        """Return an evaluated copy of the matrix"""
        result = Matrix.from_existing(self)
        result.evaluate(*args, **kwargs)
        return result

    def inverse(self):
        """Return the inverse of the matrix"""
        return Matrix.mat_inverse(self)

    def inversed(self):
        """Return an inversed copy of the matrix"""
        result = Matrix.from_existing(self)
        result.inverse()
        return result

    @staticmethod
    def mat_are_inverses(mat_a, mat_b):
        """Check if two matrices are inverses of each other"""
        if mat_a.shape != mat_b.shape:
            return False
        return np.allclose(mat_a @ mat_b, mat_a.identity())

    @staticmethod
    def mat_identity(shape):
        """Get the identity matrix of a given shape"""
        m, n = shape
        if m != n:
            raise ValueError(
                f"Invalid shape: {shape}, both 'm' and 'n' must be equal.")

        result = Matrix(np.zeros(shape))
        for i in range(m):
            result[i, i] = 1
        return result

    @staticmethod
    def mat_is_square(matrix):
        """Check if a matrix is square"""
        return matrix.m == matrix.n

    @staticmethod
    def _get_inversion_steps(shape):
        """Returns a list of points to nullify ordered in sequential order, for inversion"""
        m, n = shape

        sequence = []
        for j in range(n):
            in_triangle = False
            for i in range(m):
                if (i != j and not in_triangle):  # Wait for pivot
                    continue
                if i == j:
                    in_triangle = True
                    continue

                sequence.append((i, j))
        for j in range(n - 1, -1, -1):
            in_triangle = False
            for i in range(m - 1, -1, -1):
                if (i != j and not in_triangle):  # Wait for pivot
                    continue
                if i == j:
                    in_triangle = True
                    continue
                sequence.append((i, j))
        return sequence

    @staticmethod
    def mat_inverse(matrix) -> bool:
        """
        Attempts to calculate the inverse of the given square matrix.

        Args:
            matrix (Matrix): The matrix to be inverted.

        Returns:
            bool: True if the inversion was successful, False otherwise.

        Raises:
            TypeError: If the input matrix is not of type Matrix.
            ValueError: If the input matrix is not square.

        The function attempts to calculate the inverse of the given square matrix by performing
        row operations to nullify the values below and above the pivot. It checks for the presence
        of a non-zero pivot in each column and interchanges rows if necessary. If the matrix is
        not invertible, the function returns False. Otherwise, it normalizes the matrix pivots
        (diagonal elements) to 1's and returns True if the inverse is successfully calculated.

        Note:
            The input matrix is modified in-place if the inverse is successfully calculated.
        """
        if not isinstance(matrix, Matrix):
            raise TypeError(
                f"'matrix' must be of type {Matrix.__name__}, not {type(matrix).__name__}")
        if not Matrix.is_square(matrix):
            raise ValueError(
                f"'matrix' must be square (nxn) to conform for inversion, not {matrix.shape}")

        sequence = Matrix._get_inversion_steps(matrix.shape)
        aug_matrix = matrix.augmented_with(matrix.identity())

        # Loop through each point in the inversion sequence
        for point in sequence:
            i, j = point
            pivot = aug_matrix[j, j]

            # Check if the pivot is zero
            if pivot == 0:
                # Find a non-zero pivot in the same column
                new_pivot_index = Matrix._check_nonzero_pivot_in_col(
                    aug_matrix, j)
                if new_pivot_index != -1:
                    # Interchange rows to make the pivot non-zero
                    aug_matrix.interchange_rows(j, new_pivot_index)
                    pivot = aug_matrix[j, j]
                else:
                    # If no non-zero pivot is found, the matrix is not invertible
                    return False

            value = aug_matrix[point]
            if value == 0:
                continue

            # Perform row operations to nullify the values below and above the pivot
            aug_matrix.add_scaled_row_to_another(
                j, i, -sign(value * pivot) * abs(value / pivot))

        # Normalize the matrix pivots (diagonal to 1's)
        for i in range(aug_matrix.m):
            if aug_matrix[i, i] == 0:
                return False
            elif aug_matrix[i, i] == 1:
                continue
            aug_matrix[i] /= aug_matrix[i, i]
        inversed = aug_matrix.b_matrix

        # Check whether the inverse was actually successful
        if inversed.is_inverse_of(matrix):
            matrix[:] = inversed[:]
            return True
        return False

    @staticmethod
    def _check_nonzero_pivot_in_col(matrix, col, start_index=0):
        """Check for a non-zero pivot in a column"""
        for i in range(start_index, matrix.m):
            if matrix[i][col] != 0:
                return i
        return -1


class AugmentedMatrix(Matrix):
    """Class for working with augmented matrices"""

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
        """Returns the A component of [A | B], in this augmented matrix"""
        m, n = self.mat_a_shape
        return self[:, :n]

    @property
    def b_matrix(self):
        """Returns the B component of [A | B], in this augmented matrix"""
        m, n = self.mat_b_shape
        return self[:, -n:]
