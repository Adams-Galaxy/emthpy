from ._emthpy_rationals import Rational
from ._emthpy_functions import Function
from ._emthpy_exceptions import InvalidExpressionError

def parse_numeric(string):
    """
    Attempt to parse a string into a number (int or float).

    Args:
        string (str): The string to parse.

    Returns:    
        object: The parsed number.
    """

    if string.isnumeric():
        return int(string)
    if string.replace('.', '', 1).isnumeric():
        return float(string)
    return None


def parse_str(string):
    """
    Attempt to parse a string into a mathematical object.
    
    Args:
        string (str): The string to parse.
        
    Returns:
        object: The parsed object.
    """

    result = parse_numeric(string)
    if result is not None:
        return result
    if '/' in string:
        values = [parse_numeric(val) for val in string.split('/')]
        if None not in values and len(values) == 2:
            return Rational(*values)
    try:
        return Function(string)
    except InvalidExpressionError:
        pass
    return None
