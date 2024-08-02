"""
Module: _emthpy_function.py

This module provides classes and functions to represent and manipulate mathematical functions and expressions. 
It includes an enumeration of mathematical operators, utility functions for handling numeric conversions, 
and a `Function` class that supports various operations on mathematical expressions.

Classes:
    Operator(Enum):
        Enumeration of mathematical operators.
        
    Function:
        A class to represent a mathematical function. It supports initialization from string expressions, 
        evaluation with given variables, and conversion between infix and postfix notations.

Functions:
    try_numeric(value):
        Converts a string to a number if it is a number. Returns the converted number or the original string if it is not a number.

Class `Operator`:
    Enumeration of mathematical operators such as Sin, Cos, Tan, ASin, ACos, ATan, Sec, Cosec, Cot, LogX, Log, Ln, Power, Root, Divide, Multiply, Add, Subtract, Negative, LeftParen, and RightParen.

Class `Function`:
    Methods:
        __init__(self, expression, name='f'):
            Initialize a Function object with a mathematical expression and an optional name.
        
        __new__(cls, expression):
            Validate the expression and create a new Function object.
        
        __str__(self):
            Return the string representation of the function.
        
        __repr__(self):
            Return the string representation of the function for debugging.
        
        notation_str(self):
            Return the function notation string.
        
        __call__(self, *vars, **kwvars):
            Evaluate the function with given variables.
        
        _init_from_string(self, expression):
            Initialize the function from a string expression.
        
        evaluate(self, *vars, **kwvars):
            Evaluate the function with given variables.
        
        variables(self):
            Returns a sorted list of unique variables present in the infix expression.
        
        _validate_expression(expression):
            Validate an expression (work in progress).
        
        _is_implied_multiplication(a, b):
            Check if there is an implied multiplication between two tokens.
        
        _string_to_infix(expression):
            Convert a string expression to infix notation.
        
        _string_to_postfix(expression):
            Convert a string expression to postfix notation.

Examples:
    >>> f = Function("2x + 3")
    >>> f.name
    'f'
    >>> str(f)
    '2x + 3'
    >>> f(2)
    7
    >>> f.notation_str()
    'f(x) = 2x + 3'
    >>> f.variables()
    ['x']

This module is designed to facilitate the creation, manipulation, and evaluation of mathematical functions and expressions in a flexible and extensible manner.
"""

from enum import Enum
import numpy as np
from _emthpy_types import Vector, Matrix
from _emthpy_rationals import Rational

class Operator(Enum):
    """Enumeration of mathematical operators."""
    Sin = 0
    Cos = 1
    Tan = 2
    ASin = 3
    ACos = 4
    ATan = 5
    Sec = 6
    Cosec = 7
    Cot = 8
    LogX = 9
    Log = 10
    Ln = 11
    Power = 12
    Root = 13
    Divide = 14
    Multiply = 15
    Add = 16
    Subtract = 17
    Negative = 18
    LeftParen = 19
    RightParen = 20

# Define a dictionary of operator functions
OPERATIONS = {
    Operator.Sin: np.sin,
    Operator.Cos: np.cos,
    Operator.Tan: np.tan,
    Operator.ASin: np.arcsin,
    Operator.ACos: np.arccos,
    Operator.ATan: np.arctan,
    Operator.Sec: lambda x: 1/np.cos(x),
    Operator.Cosec: lambda x: 1/np.sin(x),
    Operator.Cot: lambda x: 1/np.tan(x),
    Operator.LogX: lambda x, y: np.log10,
    Operator.Log: np.log10,
    Operator.Ln: lambda x: np.log(x),
    Operator.Power: np.power,
    Operator.Root: np.sqrt,
    Operator.Divide: lambda x, y: x / y,
    Operator.Multiply: lambda x, y: x * y,
    Operator.Add: lambda x, y: x + y,
    Operator.Subtract: lambda x, y: x - y,
    Operator.Negative: lambda x: -x,
}

# Define the list of functions that take one argument
FUNCTIONS = [
    Operator.Sin,
    Operator.Cos,
    Operator.Tan,
    Operator.ASin,
    Operator.ACos,
    Operator.ATan,
    Operator.Sec,
    Operator.Cosec,
    Operator.Cot,
    Operator.LogX,
    Operator.Log,
    Operator.Ln,
    Operator.Root,
    Operator.Negative,
]

# Define a dictionary of operator aliases
OPERATOR_ALIASES = {
    Operator.Sin : ['sin'],
    Operator.Cos : ['cos'],
    Operator.Tan : ['tan'],
    Operator.ASin : ['asin', 'arcsin', 'sin^-1'],
    Operator.ACos : ['acos', 'arccos', 'cos^-1'],
    Operator.ATan : ['atan', 'arctan', 'tan^-1'],
    Operator.Sec : ['sec', '1/cos'],
    Operator.Cosec : ['cosec', '1/sin', 'csc'],
    Operator.Cot : ['cot', '1/tan'],
    Operator.LogX : ['logx'],
    Operator.Log : ['log'],
    Operator.Ln : ['ln'],
    Operator.Power : ['^', '**'],
    Operator.Root : ['sqrt'],
    Operator.Divide : ['/'],
    Operator.Multiply : ['*'],
    Operator.Add : ['+'],
    Operator.Subtract : ['-'],
    Operator.Negative : ['-', 'NEG'],
    Operator.LeftParen : ['('],
    Operator.RightParen : [')'],
}

# Define the characters that can be used in numeric values
NUMERIC_CHARACTERS = '0123456789.'

# Define the precedence of operators (bedmas)
PRECENDANCE = {
    Operator.Sin: 4,
    Operator.Cos: 4,
    Operator.Tan: 4,
    Operator.ASin: 4,
    Operator.ACos: 4,
    Operator.ATan: 4,
    Operator.Sec: 4,
    Operator.Cosec: 4,
    Operator.Cot: 4,
    Operator.LogX: 4,
    Operator.Log: 4,
    Operator.Ln: 4,
    Operator.Power: 3,
    Operator.Root: 3,
    Operator.Divide: 2,
    Operator.Multiply: 2,
    Operator.Negative: 2,
    Operator.Add: 1,
    Operator.Subtract: 1,
}

# Define a dictionary of mathematical constants
CONSTANTS = {'e': np.e, 'pi': np.pi,
             'inf': float('inf'), '-inf': float('-inf')}

VAR_TYPES = (
    int, float, str, Vector, Matrix
)

VALID_VAR_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'

def try_numeric(value, allow_expression=False, thow_error=False):
    """Converts a string to a number if it is a number.

    Args:
        value (str): The string to convert.

    Returns:
        int, float, or str: The converted number or the original string if it is not a number.

    Example:
        >>> try_numeric("123")
        123
        >>> try_numeric("123.45")
        123.45
        >>> try_numeric("abc")
        'abc'
    """
    if isinstance(value, (int, float)):
        return value
    formated = value.replace('.', '').replace('-', '')
    if not formated.isnumeric():
        if allow_expression:
            return Function(value)
        if thow_error:
            raise ValueError(f"Invalid number: {value}")
        return value
    return float(value) if '.' in value else int(value)

class Function:
    """A class to represent a mathematical function.
    
    This class provides methods to initialize a function from a string expression, evaluate the function with given variables,
    and convert between infix and postfix notations. It also supports operations such as string representation, function notation,
    and variable extraction.
    
    Attributes:
        name (str): The name of the function.
        _infix (list): The infix expression as a list of tokens.
        _postfix (list): The postfix expression as a list of tokens.
        
    Methods:
        __init__(self, expression, name='f'):
            Initialize a Function object with a mathematical expression and an optional name.
            
            __new__(cls, expression):
            
            __str__(self):
            
            __repr__(self):
            
            notation_str(self):
            
            __call__(self, *vars, **kwvars):
            
            _init_from_string(self, expression):
            
            evaluate(self, *vars, **kwvars):
            
            variables(self):
            
            _validate_expression(expression):
            
            _is_implied_multiplication(a, b):
            
            _string_to_infix(expression):
            
            _string_to_postfix(expression):
            
            _infix_to_postfix(expression):
            
            _evaluate_postfix(expression):
            
            Examples:
            
            >>> f = Function("2x + 3")
            
            >>> f.name
            
            'f'
            
            >>> str(f)
            
            '2x + 3'
            
            >>> f(2)
            
            7
            
            >>> f.notation_str()
            
            'f(x) = 2x + 3'
            
            >>> f.variables()
            
            ['x']
            
            This class is designed to facilitate the creation, manipulation, and evaluation of mathematical functions and expressions
            in a flexible and extensible manner.

            TODO:
                - Add rational resultant support e.g f(x) = 1/x, f(3) = 1/3 not 0.33... .
                - Add support for more complex expressions.
                - Add other initialization methods.
                - Implement validation logic.
                - Add the ability to simplify expressions.
            """

    def __init__(self, expression, name='f'):
        """Initialize a Function object.

        Args:
            expression (str): The mathematical expression.
            name (str): The name of the function. Defaults to 'f'.

        Example:
            >>> f = Function("2x + 3")
            >>> f.name
            'f'

        TODO:
            - Add support for more complex expressions.
            - Add other initialization methods.
        """
        self.expression = expression
        self.name = name
        if isinstance(expression, str):
            self._init_from_string(expression)
        else:
            raise ValueError(f"Invalid expression: {expression}, of type {type(expression).__name__}")

    def __new__(cls, expression, name='f'):
        """Initialize a Function object.

        Args:
            expression (str): The mathematical expression.
            name (str): The name of the function. Defaults to 'f'.

        Example:
            >>> f = Function("2x + 3")
            >>> f.name
            'f'
        """
        if not cls._validate_expression(expression):
            raise ValueError(f"Invalid expression: {expression}")
        return super().__new__(cls)

    def __str__(self):
        """Return the string representation of the function.

        Returns:
            str: The string representation of the function.

        Example:
            >>> str(Function("2x + 3"))
            '2x + 3'
        """
        return Function.func_to_str(self._infix)

    def __repr__(self):
        """Return the string representation of the function.

        Returns:
            str: The string representation of the function.

        Example:
            >>> str(Function("2x + 3"))
            '2x + 3'
        """
        # Replace operator aliases with their respective strings
        result = self._infix[:]
        for i, token in enumerate(result):
            if isinstance(token, Operator):
                result[i] = OPERATOR_ALIASES[token][0]
            elif isinstance(token, (int, float)):
                result[i] = str(token)
        return f"Function('{''.join(result)}')"

    def notation_str(self):
        """Return the function notation string.

        Returns:
            str: The function notation string.

        Example:
            >>> f = Function("2x + 3y")
            >>> f.notation_str()
            'f(x,y) = 2x + 3'
        """
        return f"{self.name}({','.join(self.variables())}) = {self}"

    def infix_str(self):
        """Return the infix string representation of the function.

        Returns:
            str: The infix string representation of the function.

        Example:
            >>> f = Function("2x+3")
            >>> f.infix_str()
            '2x+3'
        """
        return Function.func_to_str(self._infix)

    def postfix_str(self):
        """Return the postfix string representation of the function.

        Returns:
            str: The postfix string representation of the function.

        Example:
            >>> f = Function("2x+3")
            >>> f.postfix_str()
            '2x*3+'
        """
        return Function.func_to_str(self._postfix, False)

    def __call__(self, *vars, **kwvars):
        """Evaluate the function with given variables.

        Args:
            *vars: Positional arguments for variables.
            **kwvars: Keyword arguments for variables.

        Returns:
            float: The result of the evaluation.

        Example:
            >>> f = Function("2x + 3")
            >>> f(2)
            7
        """
        return self.evaluate(*vars, **kwvars)

    def _init_from_string(self, expression):
        """Initialize the function from a string expression.

        Args:
            expression (str): The string expression.

        Example:
            >>> f = Function("2x + 3")
            >>> f._infix
            [2, 'x', Operator.Add, 3]
        """
        self._infix = Function._string_to_infix(expression)
        Function._remove_lone_perenthisies(self._infix)
        self._postfix = Function._infix_to_postfix(self._infix)

    def evaluate(self, *vars, **kwvars):
        """Evaluate the function with given variables.

        Args:
            *vars: Positional arguments for variables.
            **kwvars: Keyword arguments for variables.

        Returns:
            float: The result of the evaluation.

        Raises:
            ValueError: If a variable is missing.

        Example:
            >>> f = Function("2x + 3")
            >>> f.evaluate(2)
            7
        """
        kwvars.update(CONSTANTS)
        substitutions = self._postfix[:]
        loose_vars = list(vars)
        loose_vars.reverse()

        for i, token in enumerate(substitutions):
            if not isinstance(token, str):
                continue
            if token in kwvars:
                if isinstance(kwvars[token], str):
                    kwvars[token] = try_numeric(kwvars[token], allow_expression=True)
                substitutions[i] = kwvars[token]
            elif len(loose_vars) > 0:
                kwvars[token] = loose_vars.pop()
                if isinstance(kwvars[token], str):
                    kwvars[token] = try_numeric(kwvars[token], allow_expression=True)
                substitutions[i] = kwvars[token]
            else:
                raise ValueError(f"Missing value for variable: {token}")
        return Function._evaluate_postfix(substitutions)

    def variables(self):
        """
        Returns a sorted list of unique variables present in the infix expression.

        This method iterates over the infix expression and identifies all the unique variables
        present in it. It then returns a sorted list of these variables.

        Returns:
            list: A sorted list of unique variables present in the infix expression.

        Example:
            >>> expression = Function("2xy - 1/z")
            >>> expression.variables()
            ['x', 'y', 'z']
        """
        result = list(set(
            [token for token in self._infix if isinstance(token, str) and 
             token not in CONSTANTS]))
        return sorted(result)

    def satisfied(self, *vars, **kwvars):
        """
        Check if the function is satisfied with the given variables.

        This method evaluates the function with the given variables and checks if the result is a number.
        If the result is a number, the function is considered satisfied.

        Args:
            *vars: Positional arguments for variables.
            **kwvars: Keyword arguments for variables.

        Returns:
            bool: True if the function is satisfied, False otherwise.

        Example:
            >>> expression = Function("2x + 3")
            >>> expression.satisfied(2)
            True
            >>> expression = Function("5x - 7y")
            >>> expression.satisfied(y=3)
            False
        """

        func_vars = self.variables()
        quick_vars = list(reversed(vars))
        for var in func_vars:
            if var in kwvars:
                print(f"{var} in kwvars")
                continue
            if len(quick_vars) > 0:
                print(f"{var} in quick_vars: {quick_vars}")
                kwvars[var] = quick_vars.pop()
            else:
                return False
        return True

    @staticmethod
    def func_to_str(func, remove_defauklt_multiplication=True):
        """
        Convert a function to a string.

        Args:
            func (Function): The function to convert.

        Returns:
            str: The string representation of the function.

        Example:
            >>> expression = Function("2x+3")
            >>> Function.func_to_str(expression)
            '2x+3'
        """
        result = func[:]

        # Remove implied multiplication operators
        if remove_defauklt_multiplication:
            for i, token in enumerate(result[1:-1]):
                if token == Operator.Multiply:
                    if Function._is_implied_multiplication(result[i], result[i+2]):
                        result[i+1] = ''

        # Replace operator aliases with their respective strings
        for i, token in enumerate(result):
            if isinstance(token, Operator):
                result[i] = OPERATOR_ALIASES[token][0]
            elif isinstance(token, (int, float)):
                result[i] = str(token)
        return ''.join(result)

    @staticmethod
    def _remove_lone_perenthisies(infix):
        """
        Remove lone parenthesis from an infix expression.

        Args:
            infix (list): The infix expression as a list of tokens.

        Example:
            >>> Function._remove_lone_perenthisies([Operator.LeftParen, 2, Operator.RightParen])
            [2]

        Notes:
            Modifies the input list in place.
        """
        result = []
        i = 0
        while i < len(infix):
            if infix[i] == Operator.LeftParen and i + 2 < len(infix) and \
                infix[i + 2] == Operator.RightParen:
                result.append(infix[i + 1])
                i += 3
            else:
                result.append(infix[i])
                i += 1
        infix.clear()
        infix.extend(result)

    @staticmethod
    def _validate_expression(expression):
        """
        (WIP)
        Validate an expression.

        Args:
            expression (str): The expression to validate.

        Returns:
            bool: True if the expression is valid, False otherwise.

        Example:
            >>> Function._validate_expression("2x + 3")
            True
            >>> Function._validate_expression("5x +")
            False

        TODO:
            - Implement validation logic.
        """

        return True

    @staticmethod
    def _is_implied_multiplication(a, b):
        """
        Check if there is an implied multiplication between two tokens.

        Args:
            a: The first token.
            b: The second token.

        Returns:
            bool: True if there is an implied multiplication, False otherwise.

        Example:
            >>> Function._is_implied_multiplication(2, 'x')
            True
        """
        return (a == Operator.RightParen or isinstance(a, (int, float, str))) and\
            (b in FUNCTIONS + [Operator.LeftParen]
            or isinstance(b, (str)))

    @staticmethod
    def _string_to_infix(expression):
        """
        Convert a string expression to infix notation.

        Args:
            expression (str): The string expression.

        Returns:
            list: The infix expression as a list of tokens.

        Example:
            >>> Function._string_to_infix("2x + 3")
            [2, 'x', Operator.Add, 3]

        TODO:
            - Functionally decompose the method.
        """
        expression = expression.replace(' ', '')
        result = []

        # Replace operator aliases with their respective enum values
        # and add '&' to the start and end of each operator (for splitting)
        for operator, aliases in OPERATOR_ALIASES.items():
            for alias in aliases:
                expression = expression.replace(alias, f"&#{operator.value}&")

        # Add '&' to the start and end of each element (for splitting)
        i = 0
        while i < len(expression):
            if expression[i] == '&': # Skip operators
                start = i
                i += 1 # Skip the first '&'
                while i < len(expression) and expression[i] != '&':
                    i += 1
                i += 1 # Skip the last '&'
                result.append(expression[start:i])
            elif expression[i] in VALID_VAR_CHARS: # Wrap variables in '&'
                start = i
                if i + 1 < len(expression) and expression[i + 1] == '_': # Handle subscript
                    while i < len(expression) and expression[i] in VALID_VAR_CHARS:
                        i += 1
                else:
                    i += 1
                result.append('&' + expression[start:i] + '&')
            elif expression[i].isnumeric(): # Wrap numbers in '&'
                start = i
                while i < len(expression) and expression[i] in NUMERIC_CHARACTERS:
                    i += 1
                result.append('&' + expression[start:i] + '&')
            else: # Add operators as they are
                result.append(expression[i])
                i += 1
        expression = ''.join(result)

        # Split the expression into tokens
        result = []
        for item in expression.split('&'):
            if item == '':
                continue
            if item[0] == '#':
                result.append(Operator(int(item[1:])))
            else:
                result.append(try_numeric(item))

        # Add implicit multiplication operators
        result2 = []
        for i, current_token in enumerate(result[:-1]):
            next_token = result[i + 1]
            result2.append(current_token)
            if Function._is_implied_multiplication(current_token, next_token):
                result2.append(Operator.Multiply)
        result2.append(result[-1])
        return result2

    @staticmethod
    def _string_to_postfix(expression):
        """
        Convert a string expression to postfix notation.
        
        Args:
            expression (str): The string expression.
            
        Returns:
            list: The postfix expression as a list of tokens.
            
        Example:
            >>> Function._string_to_postfix("2x + 3")
            [2, Operator.Multiply, 'x', 3, Operator.Add]
        """
        elements = Function._string_to_infix(expression)
        return Function._infix_to_postfix(elements)     

    @staticmethod
    def _infix_to_postfix(expression):
        """
        Convert an infix expression to postfix notation.

        Args:
            expression (list): The infix expression as a list of tokens.
        
        Returns:
            list: The postfix expression as a list of tokens.
        
        Example:
            >>> Function._infix_to_postfix(['x', Operator.Add, 3])
            ['x', 3, Operator.Add]
        """
        stack = []  # Stack to store operators
        output = []  # List to store the postfix expression

        for i, token in enumerate(expression):
            if isinstance(token, (int, float, str)):
                output.append(token)
            elif token == Operator.LeftParen:
                stack.append(token)
            elif token == Operator.RightParen:
                # Pop operators from stack and append to output until a left parenthesis is encountered
                while stack and stack[-1] != Operator.LeftParen:
                    output.append(stack.pop())
                stack.pop()  # Pop the left parenthesis
            elif token in FUNCTIONS:
                stack.append(token)
            elif token == Operator.Subtract and (i == 0 or expression[i-1] in
                [Operator.Add, Operator.Subtract, Operator.Multiply, Operator.Divide, Operator.LeftParen]):
                stack.append(Operator.Negative)
            else:  # Binary operator
                # Pop operators from stack and append to output while they have higher precedence than the current token
                while (stack and stack[-1] != Operator.LeftParen and
                    PRECENDANCE.get(stack[-1], 0) >= PRECENDANCE.get(token, 0)):
                    output.append(stack.pop())
                stack.append(token)

        # Pop any remaining operators from stack and append to output
        while stack:
            output.append(stack.pop())

        return output

    @staticmethod
    def _postfix_to_infix(expression):
        """
        (WIP) Convert a postfix expression to infix notation.

        Args:
            expression (list): The postfix expression as a list of tokens.

        Returns:
            list: The infix expression as a list of tokens.

        Example:
            >>> Function._postfix_to_infix([2, 5, Operator.Multiply, 3, Operator.Add])
            [2, 5, Operator.Multiply, 3, Operator.Add]

        TODO:
            - Properly implement bedmas
        """
        stack = []
        for token in expression:
            if isinstance(token, VAR_TYPES):
                stack.append(token)
            elif token == Operator.Negative:
                operand = stack.pop()
                stack.extend([Operator.Negative, operand])
            elif token in FUNCTIONS:
                operand = stack.pop()
                stack.extend([token, Operator.LeftParen, operand, Operator.RightParen])
            else:  # binary operators
                operand2 = stack.pop()
                operand1 = stack.pop()
                stack.extend([operand1, token, operand2])
        return stack

    @staticmethod
    def _evaluate_postfix(expression):
        """
        Evaluate a postfix expression.

        Args:
            expression (list): The postfix expression as a list of tokens.

        Returns:
            float: The result of the evaluation.

        Example:
            >>> Function._evaluate_postfix([2, 5, Operator.Multiply, 3, Operator.Add])
            13
        """
        stack = []  # Stack to store operands

        for token in expression:
            if isinstance(token, VAR_TYPES) or \
                isinstance(token, Function):
                stack.append(token)
            elif token in FUNCTIONS:
                operand = stack.pop()
                result = OPERATIONS[token](operand)
                stack.append(result)
            elif isinstance(token, Operator):
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = OPERATIONS[token](operand1, operand2)
                stack.append(result)
        return stack.pop()


    def __mul__(self, other):
        if isinstance(other, (int, float, Rational)):
            new_expression = f"({self.expression}) * {other}"
        elif isinstance(other, Function):
            new_expression = f"({self.expression}) * ({other.expression})"
        else:
            raise TypeError(
                "Cannot multiply a function by a non-numeric value")
        return Function(new_expression)

    def __rmul__(self, other):
        if isinstance(other, (int, float, Rational)):
            new_expression = f"{other} * ({self.expression})"
        elif isinstance(other, Function):
            new_expression = f"({other.expression}) * ({self.expression})"
        else:
            raise TypeError(
                "Cannot multiply a function by a non-numeric value")
        return Function(new_expression)

    def __add__(self, other):
        if isinstance(other, (int, float, Rational)):
            new_expression = f"({self.expression}) + {other}"
        elif isinstance(other, Function):
            new_expression = f"({self.expression}) + ({other.expression})"
        else:
            raise TypeError(
                "Cannot add a function to a non-numeric value")
        return Function(new_expression)

    def __radd__(self, other):
        if isinstance(other, (int, float, Rational)):
            new_expression = f"{other} + ({self.expression})"
        elif isinstance(other, Function):
            new_expression = f"({other.expression}) + ({self.expression})"
        else:
            raise TypeError(
                "Cannot add a function to a non-numeric value")
        return Function(new_expression)

    def __sub__(self, other):
        if isinstance(other, (int, float, Rational)):
            new_expression = f"({self.expression}) - {other}"
        elif isinstance(other, Function):
            new_expression = f"({self.expression}) - ({other.expression})"
        else:
            raise TypeError(
                "Cannot subtract a non-numeric value from a function")
        return Function(new_expression)

    def __rsub__(self, other):
        if isinstance(other, (int, float, Rational)):
            new_expression = f"{other} - ({self.expression})"
        elif isinstance(other, Function):
            new_expression = f"({other.expression}) - ({self.expression})"
        else:
            raise TypeError(
                "Cannot subtract a function from a non-numeric value")
        return Function(new_expression)

    def __truediv__(self, other):
        if isinstance(other, (int, float, Rational)):
            new_expression = f"({self.expression}) / {other}"
        elif isinstance(other, Function):
            new_expression = f"({self.expression}) / ({other.expression})"
        else:
            raise TypeError(
                "Cannot divide a function by a non-numeric value")
        return Function(new_expression)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, Rational)):
            new_expression = f"{other} / ({self.expression})"
        elif isinstance(other, Function):
            new_expression = f"({other.expression}) / ({self.expression})"
        else:
            raise TypeError(
                "Cannot divide a non-numeric value by a function")
        return Function(new_expression)

    def __pow__(self, other):
        if isinstance(other, (int, float, Rational)):
            new_expression = f"({self.expression}) ^ {other}"
        elif isinstance(other, Function):
            new_expression = f"({self.expression}) ^ ({other.expression})"
        else:
            raise TypeError(
                "Cannot raise a function to a non-numeric power")
        return Function(new_expression)

    def __rpow__(self, other):
        if isinstance(other, (int, float, Rational)):
            new_expression = f"{other} ^ ({self.expression})"
        elif isinstance(other, Function):
            new_expression = f"({other.expression}) ^ ({self.expression})"
        else:
            raise TypeError(
                "Cannot raise a non-numeric value to the power of a function")
        return Function(new_expression)
