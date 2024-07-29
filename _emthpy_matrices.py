"""Module for working with matrices and utilizing matrix operations"""

import numpy as np
from _emthpy_utils import sign
from _emthpy_vectors import Vector
import _emthpy_exceptions as ex
import _emthpy_utils as utils
import _emthpy_functions_DEPRICATED as func

class Matrix(np.ndarray):
    """Class for working with matrices and utilizing matrix operations"""

    def __new__(cls, *args, **kwargs):
        if "dtype" not in kwargs:
            kwargs['dtype'] = float
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
    def _from_existing(cls, matrix):
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
        """
        Check if the matrix is square.

        Returns:
            bool: True if the matrix is square, False otherwise.

        Notes:
            A matrix is square if it has the same number of rows and columns.
        """
        return Matrix.mat_is_square(self)
    def is_inverse_of(self, other):
        """
        Check if the matrix is the inverse of another matrix.

        Args:
            other (Matrix): The matrix to compare with.

        Returns:
            bool: True if the matrix is the inverse of the other matrix, False otherwise.

        Notes:
            Two matrices are inverses of each other if their product is the identity matrix.
        """
        return Matrix.mat_are_inverses(self, other)
    def augmented_with(self, other, clone_data=True):
        """
        Return an augmented matrix.

        Args:
            other (Matrix): The matrix to augment with.
            clone_data (bool, optional): Whether to clone the data of the original matrix and the other matrix. 
                Defaults to True.

        Returns:
            AugmentedMatrix: The augmented matrix.

        Notes:
            An augmented matrix is a matrix formed by appending the columns of one matrix to the columns of another matrix.
            If clone_data is True, the data of the original matrix and the other matrix are copied to the augmented matrix.
            Otherwise, the augmented matrix is created with references to the original matrices. Therefore, 
            modifying the augmented matrix will also modify the original matrices.
        """
        if clone_data:
            return AugmentedMatrix(self.copy(), other.copy())
        return AugmentedMatrix(self, other)
    def augmented_with_identity(self):
        """
        Return an augmented matrix with the identity matrix.

        Returns:
            AugmentedMatrix: The augmented matrix with the identity matrix.

        Notes:
            An augmented matrix is a matrix formed by appending the columns of one matrix to the columns of another matrix.
            This method appends the identity matrix to the original matrix.
        """
        return self.augmented_with(self.identity())
    def identity(self):
        """
        Return the identity matrix.

        Returns:
            Matrix: The identity matrix.

        Notes:
            The identity matrix is a square matrix with ones on the main diagonal and zeros elsewhere.
        """
        return Matrix.mat_identity(self.shape)
    def add_scaled_row_to_another(self, row_a, row_b, scalar):
        """
        Add a scaled row to another row.

        Args:
            row_a (int): The index of the row to be scaled and added.
            row_b (int): The index of the row to add the scaled row to.
            scalar (float): The scalar value to scale the row by.

        Notes:
            This operation modifies the matrix in-place.
        """
        self[row_b] += self[row_a] * scalar
    def scale_row(self, row, scalar):
        """
        Scale a row by a scalar.

        Args:
            row (int): The index of the row to be scaled.
            scalar (float): The scalar value to scale the row by.

        Notes:
            This operation modifies the matrix in-place.
        """
        self[row] *= scalar
    def interchange_rows(self, row_a, row_b):
        """
        Interchange two rows.

        Args:
            row_a (int): The index of the first row to be interchanged.
            row_b (int): The index of the second row to be interchanged.

        Notes:
            This operation modifies the matrix in-place.
        """
        temp = self[row_a]
        self[row_a] = self[row_b]
        self[row_b] = temp
    def evaluate(self, *args, **kwargs):
        """
        Evaluate the matrix.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Notes:
            This method evaluates any functions or rational numbers in the matrix using the given arguments.
            The evaluation is performed in-place.
        """
        for i, point in enumerate(self):
            if isinstance(self[point], (func.Function, utils.Rational)):
                self[point] = self[point].evaluate(*args, **kwargs)
    def evaluated(self, *args, **kwargs):
        """
        Return an evaluated copy of the matrix.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Matrix: The evaluated copy of the matrix.

        Notes:
            This method creates a copy of the matrix and evaluates any functions or rational numbers in the copy using the given arguments.
            The original matrix is not modified.
        """
        result = Matrix._from_existing(self)
        result.evaluate(*args, **kwargs)
        return result
    def inverse(self):
        """
        Return the inverse of the matrix.

        Returns:
            bool: True if the inversion was successful, False otherwise.

        Raises:
            TypeError: If the input matrix is not of type Matrix.
            ValueError: If the input matrix is not square.

        Notes:
            This method attempts to calculate the inverse of the given square matrix by performing row operations to nullify the values below and above the pivot.
            It checks for the presence of a non-zero pivot in each column and interchanges rows if necessary.
            If the matrix is not invertible, the method returns False.
            Otherwise, it normalizes the matrix pivots (diagonal elements) to 1's and returns True if the inverse is successfully calculated.
            The input matrix is modified in-place if the inverse is successfully calculated.
        """
        return Matrix.mat_inverse(self)
    def inversed(self):
        """
        Return an inversed copy of the matrix.

        Returns:
            Matrix: The inversed copy of the matrix.

        Notes:
            This method creates a copy of the matrix and calculates its inverse.
            The original matrix is not modified.
        """
        result = Matrix._from_existing(self)
        result.inverse()
        return result
    def transpose(self):
        """Transpose the matrix"""
        self[:] = self.T
    def transposed(self):
        """Return a transposed copy of the matrix"""
        return Matrix(self.T)
    def solve(self, RHS, raise_if_not_in_echelon=False):
        """
        Solve a matrix equation.

        Args:
            RHS (Matrix): The right-hand side of the equation.
            raise_if_not_in_echelon (bool, optional): Whether to raise an error if the matrix is not in echelon form. Defaults to False.

        Returns:
            Matrix: The solution to the matrix equation.

        Raises:
            MatrixShapeError: If the matrix is not in echelon form.

        Notes:
            This method solves a matrix equation using the given right-hand side.
            If raise_if_not_in_echelon is True, the method raises an error if the matrix is not in echelon form.
        """
        return Matrix.mat_solve(self, RHS, raise_if_not_in_echelon)
    def reduce_to_echelon_form(self):
        """Convert the matrix to echelon form"""
        self[:] = Matrix.mat_to_echelon_form(self)[:]
    def is_in_echelon_form(self):
        """Check if the matrix is in echelon form"""
        return Matrix.mat_is_in_echelon_form(self)

    @staticmethod
    def mat_to_echelon_form(matrix):
        """Convert a matrix to echelon form"""
        result = matrix.copy()
        m, n = result.shape
        pivot = 0
        for j in range(n):
            found = False
            for i in range(pivot, m):
                if result[i, j] != 0:
                    found = True
                    result.interchange_rows(i, pivot)
                    break
            if found:
                for i in range(pivot + 1, m):
                    result.add_scaled_row_to_another(
                        pivot, i, -result[i, j] / result[pivot, j])
                pivot += 1
        return result
    @staticmethod
    def mat_is_in_echelon_form(matrix):
        """Check if a matrix is in echelon form"""
        points = Matrix._lower_triangle(matrix.shape)
        return all(matrix[point] == 0 for point in points)
    @staticmethod
    def mat_dot(mat_a, mat_b):
        """Dot product of two matrices"""
        if mat_a.n != mat_b.m:
            raise ex.MatrixShapeError(
                f"Cannot multiply matrices of shape {mat_a.shape} and {mat_b.shape}")
        return Matrix(np.dot(mat_a, mat_b))
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

        aug_matrix = matrix.augmented_with_identity()

        # Loop through each point in the inversion sequence
        m, n = matrix.shape
        pivot = 0
        for j in range(n):
            found = False
            for i in range(pivot, m):
                if aug_matrix[i, j] != 0:
                    found = True
                    aug_matrix.interchange_rows(i, pivot)
                    break
            if found:
                for i in range(pivot + 1, m):
                    aug_matrix.add_scaled_row_to_another(
                    pivot, i, -aug_matrix[i, j] / aug_matrix[pivot, j])
                for i in range(pivot - 1, -1, -1):
                    aug_matrix.add_scaled_row_to_another(
                    pivot, i, -aug_matrix[i, j] / aug_matrix[pivot, j])
            pivot += 1
        inversed = aug_matrix.b_matrix # Extract the inverse matrix

        # Check whether the inverse was successful
        if inversed.is_inverse_of(matrix):
            matrix[:] = inversed[:] # Modify the original matrix
            return True
        return False 
    @staticmethod
    def mat_solve(matrix, RHS, raise_if_not_in_echelon=False):
        """Solve a matrix equation"""
        if raise_if_not_in_echelon and not Matrix.mat_is_in_echelon_form(matrix):
            raise ex.MatrixShapeError(
                "Matrix must be in echelon form to solve an equation.")
        if not matrix.is_square():
            raise ex.MatrixShapeError(
                "Matrix must be square (Equal piviots and variables) to solve an equation.")
        return np.linalg.solve(matrix, RHS)
    @staticmethod
    def _lower_triangle(shape):
        """Returns a list of points to nullify ordered in sequential order, for lower triangle"""
        m, n = shape
        sequence = []
        for j in range(n):
            for i in range(j + 1, m):
                sequence.append((i, j))
        return sequence
    @staticmethod
    def _upper_triangle(shape):
        """Returns a list of points to nullify ordered in sequential order, for upper triangle"""
        m, n = shape
        sequence = []
        for j in range(n - 1, -1, -1):
            for i in range(j - 1, -1, -1):
                sequence.append((i, j))
        return sequence

    # Replace numpys default array multiplication with matrix multiplication
    def __mul__(self, other):
        if isinstance(other, Matrix):
            return super().__matmul__(other)
        return super().__mul__(other)
    def __rmul__(self, other):
        if isinstance(other, Matrix):
            return super().__matmul__(other)
        return super().__mul__(other)
 

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
