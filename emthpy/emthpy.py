import numpy as np
from .package._emthpy_types import *

"""Contains data structures for Engineering Mathematics"""

def matrix(*args, default_dynamic=True, **kwargs):
    """Return a matrix object"""
    return Matrix(*args, **kwargs)

def vector(*args, **kwargs):
    """Return a vector object"""
    return Vector(*args, **kwargs)

def cross(a, b, *args, **kwargs):
    """Return the cross product of two vectors"""
    return Vector.vec_cross(a, b, *args, **kwargs)

def function(*args, **kwargs):
    """Return an equation object"""
    return Function(*args, **kwargs)

def rational(a, b, *args, **kwargs):
    """Return a fraction object"""
    return Rational(a, b, *args, **kwargs)