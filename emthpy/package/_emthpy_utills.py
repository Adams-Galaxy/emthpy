"""This module contains utility functions for the emthpy package"""

from typing import Union


def is_whole_number(x):
    """Check if a number is a whole number"""
    return x == x // 1


def clean_numeric(x: float) -> Union[int, float]:
    """Return a whole number if the number is whole"""
    if isinstance(x, int):
        return x
    if not isinstance(x, float):
        raise TypeError(f"Expected float, got {type(x).__name__}")
    return int(x) if is_whole_number(x) else x

def sign(x):
    """Return the sign of a number"""
    return int(x / (x // 1)) if x != 0 else 1
