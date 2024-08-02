import _emthpy_function as func
import _emthpy_vectors as vec
import _emthpy_matrices as mat
import _emthpy_rationals as rat

"""Contains data structures for Engineering Mathematics"""

def matrix(*args, **kwargs):
    """Return a matrix object"""
    return mat.Matrix(*args, **kwargs)

def vector(*args, **kwargs):
    """Return a vector object"""
    return vec.Vector(*args, **kwargs)

def function(*args, **kwargs):
    """Return an equation object"""
    return func.Function(*args, **kwargs)

def fraction(a, b, *args, **kwargs):
    """Return a fraction object"""
    return rat.Rational(a, b, *args, **kwargs)