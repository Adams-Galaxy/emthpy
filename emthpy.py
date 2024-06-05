import _emthpy_equations as eq
import _emthpy_vectors as vec
import _emthpy_matrices as mat
import _emthpy_utils as utils
from _emthpy_command_line import run_command_line

"""Contains data structures for Engineering Mathematics"""

def matrix(*args, **kwargs):
    """Return a matrix object"""
    return mat.Matrix(*args, **kwargs)

def vector(*args, **kwargs):
    """Return a vector object"""
    return vec.Vector(*args, **kwargs)

def function(*args, **kwargs):
    """Return an equation object"""
    return eq.Equation(*args, **kwargs)

def fraction(a, b, *args, **kwargs):
    """Return a fraction object"""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return utils.FFraction(a, b)
    return utils.Fraction(a, b, *args, **kwargs)