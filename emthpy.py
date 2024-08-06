import numpy as np
import _emthpy_functions as _func
import _emthpy_vectors as _vec
import _emthpy_matrices as _mat
import _emthpy_rationals as _rat


"""Contains data structures for Engineering Mathematics"""

def matrix(*args, default_dynamic=True, **kwargs):
    """Return a matrix object"""
    if isinstance(args[0], np.ndarray):
        return _mat.Matrix(*args, **kwargs)
    if default_dynamic:
        return _mat.DMatrix(*args, **kwargs)

    array = []
    for row in args[0]:
        array.extend(row)

    # Check if the matrix contains any non-numeric values
    if True in [not isinstance(i, (int, float)) for i in array]:
        return _mat.DMatrix(*args, **kwargs)
    return _mat.Matrix(*args, **kwargs)

def vector(*args, **kwargs):
    """Return a vector object"""
    return _vec.Vector(*args, **kwargs)

def cross(a, b, *args, **kwargs):
    """Return the cross product of two vectors"""
    return _vec.Vector.vec_cross(a, b, *args, **kwargs)

def function(*args, **kwargs):
    """Return an equation object"""
    return _func.Function(*args, **kwargs)

def fraction(a, b, *args, **kwargs):
    """Return a fraction object"""
    return _rat.Rational(a, b, *args, **kwargs)