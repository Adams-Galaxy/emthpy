"""Module for working with matrices and utilizing matrix operations"""

import numpy as np
from _emthpy_vectors import Vector, i, j, k
import _emthpy_exceptions as ex
from _emthpy_rationals import Rational
from _emthpy_functions import Function

class Matrix(np.ndarray):
    """Class for working with matrices and utilizing matrix operations"""

    MAX_PRINT_SIZE = 5

    def __new__(cls, *args, **kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = float
        obj = np.asarray(*args, **kwargs).view(cls)
        if len(obj.shape) > 2:
            raise ex.MatrixShapeError(
                f"matrix can only be of shape (m, n), not {obj.shape}")
        return obj
    def __array_finalize__(self, obj):
        if obj is None:
            return

    @classmethod
    def _from_existing(cls, matrix):
        """Create a matrix from an existing matrix"""
        return matrix.copy().view(cls)

    @staticmethod
    def matrix_to_str(matrix, max_size=MAX_PRINT_SIZE):
        """
        Convert a matrix to a string.

        Args:
            matrix (np.ndarray): The matrix to convert to a string.
            max_size (int, optional): The maximum size of the string. Defaults to Matrix.MAX_PRINT_SIZE.
        """
        str_matrix = np.array(matrix, dtype=str)
        # Truncate long strings
        if max_size > 0:
            for point in np.ndindex(matrix.shape):
                if len(str_matrix[point]) > Matrix.MAX_PRINT_SIZE:
                    str_matrix[point] = str_matrix[point][:5] + '...'

        # Determine the maximum width of each column
        col_widths = [max(len(item) for item in col) for col in str_matrix.T]

        # Format each row
        rows = []
        for row in str_matrix:
            formatted_row = " ".join(
                f"{item:<{col_widths[i]}}" for i, item in enumerate(row))
            rows.append('[' + formatted_row + ']')

        # Join all rows into a single string
        return "\n".join(rows)

    def __str__(self) -> str:
        return Matrix.matrix_to_str(self)

    @property
    def m(self):
        """Return the number of rows in the matrix"""
        return self.shape[0]
    @property
    def n(self):
        """Return the number of columns in the matrix"""
        return self.shape[1]
    @property
    def determinant(self):
        """The determinant of the matrix"""
        return Matrix.mat_determinant(self)

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
    def add_scale_divided_row_to_another(self, row_a, row_b, scalar):
        """
        Add a scaled row to another row.

        Args:
            row_a (int): The index of the row to be scaled and added.
            row_b (int): The index of the row to add the scaled row to.
            scalar (float): The scalar value to scale the row by.

        Notes:
            This operation modifies the matrix in-place.
        """
        self[row_b] += self[row_a] / scalar
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
    def scale_divide_row(self, row, scalar):
        """
        Scale a row by the inverse of a scalar.

        Args:
            row (int): The index of the row to be scaled.
            scalar (float): The scalar value to divide the row by.

        Notes:
            This operation modifies the matrix in-place.
        """
        self[row] /= scalar
    def interchange_rows(self, row_a, row_b):
        """
        Interchange two rows.

        Args:
            row_a (int): The index of the first row to be interchanged.
            row_b (int): The index of the second row to be interchanged.

        Notes:
            This operation modifies the matrix in-place.
        """
        temp = self[row_a, :].copy()
        self[row_a, :] = self[row_b, :]
        self[row_b, :] = temp[:]
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
    def inversable(self):
        """Check if the matrix is inversable"""
        return self.is_square() and self.determinant != 0
    def transpose(self):
        """Transpose the matrix"""
        self[:] = self.T
    def transposed(self):
        """Return a transposed copy of the matrix"""
        return Matrix(self.T)
    def det(self):
        """Return the determinant of the matrix"""
        return Matrix.mat_determinant(self)
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
    def in_echelon_form(self):
        """Check if the matrix is in echelon form"""
        return Matrix.mat_is_in_echelon_form(self)

    def copy(self):
        """
        Return a copy of the matrix
        
        Returns:
            Matrix: A copy of the matrix.
        """
        return super().copy().view(type(self))
        

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

        result = Matrix(np.zeros(shape), dtype=int)
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
        if matrix.determinant == 0:
            return False

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
                if aug_matrix[pivot, j] != 1:
                    aug_matrix.scale_divide_row(pivot, aug_matrix[pivot, j])
            pivot += 1
        inversed = aug_matrix.b_matrix # Extract the inverse matrix
        if matrix.dtype == int or matrix.dtype == float and inversed.dtype == object:
            inversed.evaluate() # Evaluate the inverse matrix (convert rationals to numerical values)

        matrix[:] = inversed[:] # Modify the original matrix
        return True
    
    @staticmethod
    def mat_determinant(matrix):
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
                f"'matrix' must be square (nxn) to conform for determination, not {matrix.shape}")

        upper_triangular = matrix.copy()

        # Loop through each point in the inversion sequence
        m, n = matrix.shape
        pivot = 0
        for j in range(n):
            found = False
            for i in range(pivot, m):
                if upper_triangular[i, j] != 0:
                    found = True
                    upper_triangular.interchange_rows(i, pivot)
                    break
            if found:
                for i in range(pivot + 1, m):
                    upper_triangular.add_scaled_row_to_another(
                    pivot, i, -upper_triangular[i, j] / upper_triangular[pivot, j])
            pivot += 1

        # Calculate the determinant
        det = np.prod([upper_triangular[i, i] for i in range(m)])
        return det
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



class DMatrix(Matrix):
    """
    Dynamic Matrix (DMatrix) class for performing matrix operations.

    This class provides a flexible and efficient way to handle matrices, 
    supporting various matrix operations such as addition, multiplication, 
    transposition, and more. It is designed to work seamlessly with numpy arrays.

    Attributes:
        data (np.ndarray): The underlying data of the matrix.
        shape (tuple): The shape of the matrix (rows, columns).

    Methods:
        
        transpose(self):
            Return the transpose of the matrix.
        
        determinant(self):
            Return the determinant of the matrix.
        
        inverse(self):
            Return the inverse of the matrix.

    Examples:
        >>> m1 = DMatrix([[1, 2], [3, 4]])
        >>> m2 = DMatrix([[5, 6], [7, 8]])
        >>> m3 = m1 + m2
        >>> print(m3)
        [[ 6  8]
         [10 12]]
        
        >>> m5 = m1.transpose()
        >>> print(m5)
        [[1 3]
         [2 4]]
        
        >>> det = m1.determinant()
        >>> print(det)
        -2.0
        
        >>> inv = m1.inverse()
        >>> print(inv)
        [[-2.   1. ]
         [ 1.5 -0.5]]

    Notes:
        - The class assumes that the input data is a valid 2D array-like structure.
        - The class provides basic error handling for invalid operations.
    """
    
    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the DMatrix class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            DMatrix: The new instance of the DMatrix class.

        Notes:
            This method creates a new instance of the DMatrix class. It accepts any data type as input and converts it to the appropriate type for the matrix elements. If the input is a string, it checks if it represents a rational number or a function and converts it accordingly. If the input is not a rational number or a function, it is converted to a Rational number with a denominator of 1.

        Raises:
            MatrixShapeError: If the matrix shape is not valid.

        """
        if 'dtype' in kwargs:
            obj = np.asarray(*args, **kwargs).view(cls)
        else:
            kwargs['dtype'] = object
            obj = np.asarray(*args, **kwargs).view(cls)
            for point in np.ndindex(obj.shape):
                if isinstance(obj[point], str):
                    split = obj[point].split('/')
                    if len(split) == 2 and split[0].isnumeric() and split[1].isnumeric():
                        obj[point] = Rational(int(split[0]), int(split[1]))
                    else:
                        obj[point] = Function(obj[point])
                    continue
                if not isinstance(obj[point], Rational):
                    obj[point] = Rational(obj[point], 1)
        if len(obj.shape) > 2:
            raise ex.MatrixShapeError(
                f"matrix can only be of shape (m, n), not {obj.shape}")
        return obj
    
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
        for point in np.ndindex(self.shape):
            if callable(self[point]):  # Evaluate functions and rationals
                self[point] = self[point](*args, **kwargs)

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
        result = self.copy().view(Matrix)
        result.evaluate(*args, **kwargs)
        return result

    def evaluate_expressions(self, *args, **kwargs):
        """
        Evaluate the matrix expressions.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Notes:
            This method evaluates any functions or expressions in the matrix using the given arguments.
            The evaluation is performed in-place.

        """
        for point in np.ndindex(self.shape):
            if isinstance(self[point], Function):
                self[point] = self[point](*args, **kwargs)

    def evaluated_expressions(self, *args, **kwargs):
        """
        Return an evaluated copy of the matrix expressions.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Matrix: The evaluated copy of the matrix expressions.

        Notes:
            This method creates a copy of the matrix and evaluates any functions or expressions in the copy using the given arguments.
            The original matrix is not modified.

        """
        result = self.copy()
        result.evaluate_expressions(*args, **kwargs)
        return result

    def __call__(self, *args, **kwargs):
        return self.evaluated_expressions(*args, **kwargs)


class AugmentedMatrix(Matrix):
    """Class for working with augmented matrices"""

    def __new__(cls, mat_a, mat_b, **kwargs):
        if not mat_a.dtype in [int, float] or not mat_b.dtype in [int, float]:
            kwargs['dtype'] = object

        obj = super().__new__(cls, np.hstack((mat_a, mat_b)), **kwargs)
        return obj

    def __array_finalize__(self, obj):
        return super().__array_finalize__(obj)

    def __init__(self, mat_a, mat_b, **kwargs) -> None:
        super().__init__(**kwargs)

        self.mat_a_shape = mat_a.shape
        self.mat_b_shape = mat_b.shape

        self.mat_a_type = type(mat_a)
        self.mat_b_type = type(mat_b)

    @property
    def a_matrix(self):
        """Returns the A component of [A | B], in this augmented matrix"""
        m, n = self.mat_a_shape
        return self[:, :n].view(self.mat_a_type)

    @property
    def b_matrix(self):
        """Returns the B component of [A | B], in this augmented matrix"""
        m, n = self.mat_b_shape
        return self[:, -n:].view(self.mat_b_type)

