import math
from enum import Enum
import numpy as np
import _emthpy_exceptions as ex

# Define a string of numeric characters
NUMERIC_CHARS = "0123456789."

def isnumeric(c: str) -> bool:
    """Check if a character is numeric."""
    return c in NUMERIC_CHARS

def is_basic_operator(c: str) -> bool:
    """Check if a character is a basic operator (+, -, *, /, ^)."""
    return c in "+-*/^"

def isvarchar(c: str) -> bool:
    """Check if a character is a valid variable character."""
    return c in "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def find_all(s, substring):
    """Find all occurrences of a substring in a string."""
    return [i for i, char in enumerate(s) if s[i:i+len(substring)] == substring]


class Operator(Enum):
    """Enumeration of mathematical operators."""
    Sin = 'sin'
    Cos = 'cos'
    Tan = 'tan'
    ASin = 'asin'
    ACos = 'acos'
    ATan = 'atan'
    Sec = 'sec'
    Cosec = 'cosec'
    Cot = 'cot'
    LogX = 'logx'
    Log = 'log'
    Ln = 'ln'
    Power = '^'
    Root = 'sqrt'
    Divide = '/'
    Multiply = '*'
    Add = '+'
    Subtract = '-'
    Negative = '--'
    Dot = '.'

    def placeholder(self):
        """Get the placeholder string for the operator."""
        return "&" + self.value + "&"
    
    def ismodifier(self):
        """Check if the operator is a modifier (takes one argument)."""
        return len(OPERATOR_SIGNATURES[self]) == 1

    def preform(self, *args):
        """Perform the operation with the given arguments."""
        return OPERATIONS[self](*args)

    def __repr__(self):
        """Return the string representation of the operator."""
        return f"{Operator.__name__}.{self.name}: \'{self.value}\'"

    def __str__(self):
        """Return the string representation of the operator."""
        key = OPERATOR_KEYS[self]
        return key[0] if isinstance(key, list) else key

# Define a dictionary of mathematical constants
CONSTANTS = {'e': math.e, 'pi': math.pi, 'inf': float('inf'), '-inf': float('-inf')}

# Define the order of operations (BEDMAS)
BEDMAS = (
    (Operator.Negative, Operator.Dot),
    (Operator.Sin, Operator.Cos, Operator.Tan, Operator.ASin, Operator.ACos, Operator.ATan,
     Operator.Sec, Operator.Cosec, Operator.Cot, Operator.LogX, Operator.Log, Operator.Ln),
    (Operator.Power, Operator.Root),
    (Operator.Multiply, Operator.Divide),
    (Operator.Add, Operator.Subtract)
)

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
    Operator.Dot: np.dot
}

# Define a dictionary of operator keys
OPERATOR_KEYS = {
    Operator.ASin: ['asin', 'sin^-1'],
    Operator.ACos: ['acos', 'cos^-1'],
    Operator.ATan: ['atan', 'tan^-1'],
    Operator.Sec: 'sec',
    Operator.Cosec: 'cosec',
    Operator.Cot: 'cot',
    Operator.Sin: 'sin',
    Operator.Cos: 'cos',
    Operator.Tan: 'tan',
    Operator.Ln: ['ln', 'log'],
    Operator.Log: 'log10',
    Operator.Power: ['^', '**'],
    Operator.Root: 'sqrt',
    Operator.Add: '+',
    Operator.Subtract: '-',
    Operator.Multiply: '*',
    Operator.Divide: '/',
    Operator.Dot: '.',
    Operator.Negative: '-'
}

OPERATOR_REQUIREMENTS = {
    Operator.Dot: lambda a, b: not isnumeric(a) and not isnumeric(b),
}

OPERATOR_SIGNATURES = {
    Operator.ASin: (1,),
    Operator.ACos: (1,),
    Operator.ATan: (1,),
    Operator.Sec: (1,),
    Operator.Cosec: (1,),
    Operator.Cot: (1,),
    Operator.Sin: (1,),
    Operator.Cos: (1,),
    Operator.Tan: (1,),
    Operator.Ln: (1,),
    Operator.Log: (1,),
    Operator.Power: (-1, 1),
    Operator.Root: (1,),
    Operator.Add: (-1, 1),
    Operator.Subtract: (-1, 1),
    Operator.Multiply: (-1, 1),
    Operator.Divide: (-1, 1),
    Operator.Dot: (-1, 1),
    Operator.Negative: (1,)
}


class VariableSet(dict):
    """Dictionary subclass for storing variables."""
    def __init__(self, *args, prompt_on_undefinged=False, **kwargs):
        self.prompt_on_undefinged = prompt_on_undefinged
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if value is None:
            if self.prompt_on_undefinged:
                return float(input(str(key + ": ")))
            raise ex.UndefingedVariableError(f"Variable '{key}' is not defined")
            
        return super().__getitem__(key)

    def update_vars(self, *args, raise_if_mismatched=False, **kwargs):
        """Update the variable values."""
        # Assign values to variables if provided as arguments
        if len(args) > 0:
            if len(args) != len(self) and raise_if_mismatched:
                raise ex.InvalidArgumentError("Invalid number of arguments (" +
                                              f"{len(args)}) for {len(self)} variables")
            for i, var in enumerate(self.keys()):
                if i >= len(args):
                    break
                if isinstance(args[i], str):
                    self[var] = Function(args[i])
                else:   self[var] = args[i]

        # Assign values to variables if provided as keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, str):
                self[key] = Function(value)
            else:   self[key] = value

    def defined(self, var):
        """Check if a variable is defined within the variable set."""
        if var not in self:
            return False
        return super().__getitem__(var) is not None

    def copy(self):
        """Return a copy of the VariableSet object."""
        return VariableSet(self, prompt_on_undefinged=self.prompt_on_undefinged)
        
class Function:
    """Class representing a mathematical equation."""

    def __new__(cls, equation=..., *args, variable_set=..., existing=...,
                require_format=True, **kwargs):
        """Create an Equation object."""

        if not Function.is_valid(equation):
            raise ex.InvalidEquationError(f"Invalid equation: \"{equation}\"")
        return super().__new__(cls)

    def __init__(self, equation=..., *args, variable_set=..., existing=...,
                 require_format=True, **kwargs):
        self.name = kwargs.get('name', 'f')
        if existing != ...:
            self._init_from_existing(existing, *args, **kwargs)
            return

        # Initialize the Equation object
        if variable_set != ...:
            self.variable_set = variable_set
        else:
            self.variable_set = VariableSet()

        if isinstance(equation, str):
            self._init_from_str(equation, *args, **kwargs)
        elif isinstance(equation, list):
            self.equation = equation
            if require_format:
                self.equation = Function.format(equation)
        self.update_variables(*args, **kwargs)

    def _init_from_str(self, equation, *args, **kwargs):
        """Initialize the Equation object from a string."""
        
        # Refactor the equation string to replace functions with single char identifiers
        equation = self.refactor_str_operators(equation + '\r')

        # Interpret the refactored equation string
        self.unformated = self.interperate(equation)

        # Format the interpreted equation
        self.equation = Function.format(self.unformated)

        self.update_variables(*args, **kwargs)
    def _init_from_existing(self, equation, *args, **kwargs):
        """Initialize the Equation object from an existing Equation object."""
        self.equation = equation.equation[:]
        self.unformated = equation.unformated[:]
        self.variable_set = equation.variable_set
        self.update_variables(*args, **kwargs)

    @classmethod
    def from_existing(cls, equation, *args, **kwargs):
        """Create an Equation object from an existing Equation object."""
        return cls(existing=equation, *args, **kwargs)
    @classmethod
    def from_numeric(cls, value, *args, **kwargs):
        """Create an Equation object from a numeric value."""
        return cls(existing=[value], *args, **kwargs)
    def _decomposed(self):
        """Decompose the equation into its components."""
        def decompose(equation):
            result = []
            for item in equation:
                if isinstance(item, list):
                    result.extend(decompose(item))
                else:
                    result.append(item)
            return result
        result = decompose(self.equation)
        return result
    def function_variables(self):
        """
        Get the variables in the equation.

        Returns:
            tuple: A tuple containing the variables in the equation.
        """
        vars = []
        for item in self._decomposed():
            if isinstance(item, str) and item \
                not in vars and item not in CONSTANTS:
                vars.append(item)
        return tuple(vars)

    def _remove_lone_parentheses(self):
        """
        Remove lone parentheses from the equation.
        """
        def remove_parentheses(equation):
            if len(equation) == 1:
                return equation[0]
            for i, item in enumerate(equation):
                if isinstance(item, list):
                    equation[i] = remove_parentheses(item)
            return equation
        self.equation = remove_parentheses(self.equation)
    def _requires_format(self):
        """
        Check if the equation requires formatting.

        Returns:
            bool: True if the equation requires formatting, False otherwise.
        """
        return True in [isinstance(item, list) for item in self.equation]
    def update_variables(self, *args, **kwargs):
        """
        Update the variable values.

        Args:
            *args: Variable arguments.
            **kwargs: Keyword arguments.
        """
        self.variable_set.update_vars(*args, **kwargs)
    def evaluate(self, *args, **kwargs):
        """
        Solve the equation with the given variable values.

        Args:
            *args: Variable arguments.
            **kwargs: Keyword arguments.

        Returns:
            float: The result of the equation evaluation.
        """
        # Update the variables if arguments are provided
        var_set = self.variable_set.copy()
        var_set.update_vars(*args, **kwargs)
        return Function._eq_evaluate(self.equation, var_set)
    def evaluate_from(self, a, b):
        """
        Solve the equation with the given variable values.

        Args:
            a: The starting variable values.
            b: The ending variable values.

        Returns:
            float: The result of the equation evaluation.
        """
        if isinstance(a, tuple) and isinstance(b, tuple):
            return self.evaluate(*b[0], **b[1]) - self.evaluate(*a[0], **a[1])
        return self.evaluate(b) - self.evaluate(a)
    def refactor_str_operators(self, equation: str) -> str:
        """
        Replace functions in the equation string with single char identifiers.

        Args:
            equation (str): The equation string to be refactored.

        Returns:
            str: The refactored equation string.
        """
        def replace_operator(operator, replacement, equation):
            prev_index = 0
            index = equation.find(replacement, prev_index)
            while index != -1:
                if operator in OPERATOR_REQUIREMENTS:
                    a = equation[index - 1] if index > 0 else ''
                    b = equation[index + 1] if index < len(equation) - 1 else ''
                    if not OPERATOR_REQUIREMENTS[operator](a, b):
                        prev_index = index + 1
                        index = equation.find(replacement, prev_index)
                        continue

                equation = equation[:index] + \
                    operator.placeholder() + \
                    equation[index + len(replacement):]
                prev_index = index + len(replacement) + 2
                index = equation.find(replacement, prev_index)
            return equation

        # Replace functions (e.g sin) with single char identifiers
        for operator, identifier in OPERATOR_KEYS.items():
            if operator is Operator.Negative:
                continue
            if isinstance(identifier, list):
                for value in identifier:
                    equation = replace_operator(operator, value, equation)
            else:
                equation = replace_operator(operator, identifier, equation)

        for key, value in CONSTANTS.items():
            equation = equation.replace(key, f"!{key}!")
        return equation
    def interperate(self, equation: str) -> list:
        """
        Interprets the equation string and converts it into a list of tokens.

        Args:
            equation (str): The equation string to be interpreted.

        Returns:
            list: The list of tokens representing the interpreted equation.
        """
        equation = equation.replace(' ', '')
        result = []
        last_result = None

        skip_to = 0
        pending_operator = False
        for i, c in enumerate(equation):
            if skip_to > i or c == '\r':
                continue

            if len(result) > 0:
                last_result = result[len(result) - 1]

            if isnumeric(c):
                numeric_values = ""
                for e in range(len(equation) - i):
                    if not isnumeric(equation[i + e]):
                        break
                    numeric_values += equation[i + e]

                result.append(float(numeric_values) if '.' \
                    in numeric_values else int(numeric_values))
                pending_operator = False
                skip_to = i + len(numeric_values)
            elif isvarchar(c):
                # Check for implied multiplication
                if Function._is_default_multiplier(last_result):
                    if not pending_operator:
                        result.append(Operator.Multiply)
                alpha_values = Function.find_var_in_str(equation[i:])
                result.append(alpha_values)

                self.variable_set[alpha_values] = None
                pending_operator = False
                skip_to = i + len(alpha_values)
            elif c == '(':
                endpos = Function._find_end_bracket(equation, i+1)
                substring = equation[i+1:endpos]
                skip_to = endpos + 1
                # Check for implied multiplication
                if Function._is_default_multiplier(last_result):
                        if not pending_operator:
                            result.append(Operator.Multiply)
                result.append(self.interperate(substring))
            elif c == '&':
                endpos = equation.find('&', i + 1)
                substring = equation[i+1:endpos]
                skip_to = endpos + 1

                if substring == '-':
                    op = Operator.Negative if i == 0 or pending_operator \
                        else Operator.Subtract
                else:
                    op = Operator(substring)
                result.append(op)

                if op is not Operator.Negative:
                    pending_operator = True
            elif c == '!':
                endpos = equation.find('!', i + 1)
                substring = equation[i+1:endpos]
                skip_to = endpos + 1

                # Check for implied multiplication
                if Function._is_default_multiplier(last_result):
                    if not pending_operator:
                        result.append(Operator.Multiply)
                result.append(substring)
            else:
                raise ex.UnkownOperatorError(f"Unkown operator: \'{c}\'")
        return result
    def trap_intergral(self, a, b, n=-1, accuracy_factor=10):
        """
        Trapazoidal intergral approximation.

        Args:
            a: The starting point of the interval.
            b: The ending point of the interval.
            n (int, optional): The number of subintervals. Defaults to -1.
            accuracy_factor (int, optional): The factor to determine the number of subintervals. Defaults to 10.

        Returns:
            float: The approximate value of the integral.
        """
        if n == -1:
            n = math.ceil((b - a) * accuracy_factor)

        h = (b - a) / n
        result = (self(a) + self(b))/2 + sum([self(a + i*h) for i in range(1, n)])
        return result * h
    def simps_intergral(self, a, b, n=-1, accuracy_factor=10):
        """
        Simpson's rule intergral approximation.

        Args:
            a: The starting point of the interval.
            b: The ending point of the interval.
            n (int, optional): The number of subintervals. Defaults to -1.
            accuracy_factor (int, optional): The factor to determine the number of subintervals. Defaults to 10.

        Returns:
            float: The approximate value of the integral.
        """
        if n == -1:
            n = math.ceil((b - a) * accuracy_factor)
        if n % 2 != 0:
            n += 1

        h = (b - a) / n
        result = self(a) + self(b) + 4 * sum([self(a + i*h) for i in range(1, n, 2)]) + \
            2 * sum([self(a + i*h) for i in range(2, n, 2)])
        return result * h / 3
    def sort_vars_by_occurance(self):
        """
        Sort the variables by occurrence in the equation.
        """
        vars = []
        for item in self._decomposed():
            if isinstance(item, str) and item not in vars:
                vars.append(item)

        result = {key: self.variable_set[key] for key in vars}
        result.update(self.variable_set)
        self.variable_set.clear()
        self.variable_set.update(result)
    def contains_all(self, vars):
        """
        Check if the equation contains all the specified variables.

        Args:
            vars: The variables to check.

        Returns:
            bool: True if the equation contains all the variables, False otherwise.
        """
        if isinstance(vars, str):
            return vars in self.function_variables()
        if len(vars) == 0:
            return False
        included_vars = self.function_variables()
        for var in vars.keys():
            if var not in included_vars:
                return False
        return True
    def contains_any(self, vars):
        """
        Check if the equation contains any of the specified variables.

        Args:
            vars: The variables to check.

        Returns:
            bool: True if the equation contains any of the variables, False otherwise.
        """
        if isinstance(vars, str):
            return vars in self.function_variables()
        if len(vars) == 0:
            return False
        included_vars = self.function_variables()
        for var in vars.keys():
            if var in included_vars:
                return True
        return False
    def vars_satisfied(self, *args, **kwargs):
        """
        Check if all the variables in the equation are satisfied.

        Args:
            *args: Variable arguments.
            **kwargs: Keyword arguments.

        Returns:
            bool: True if all the variables are satisfied, False otherwise.

        Note:
            The function will return False if any of the variables are not defined.
        """
        temp_var_set = self.variable_set.copy()
        temp_var_set.update_vars(*args, **kwargs)
        for var in temp_var_set.keys():
            if not temp_var_set.defined(var):
                return False
        return True

    @staticmethod
    def is_valid(function) -> bool:
        """(WIP) Check if the equation string is valid."""
        if not isinstance(function, str):
            return False

        return True
    @staticmethod
    def find_var_in_str(str):
        """Find a variable in a string."""
        if len(str) == 1 or str[1] != '_':
            return str[0]
        
        result = ""
        for c in str:
            if isvarchar(c):
                result += c
            else:
                return result
        return result    
    @staticmethod
    def format(equation: list) -> list:
        """Format the equation list."""
        result = equation[:]
                    
        for i, item in enumerate(result):
            if isinstance(item, list):
                result[i] = Function.format(item)
        
        for opp in BEDMAS:
            for i, item in enumerate(result):
                if item in opp:
                    if item.ismodifier():
                        result[i] = [item, Function._cut_next_item(result, i)]
                        continue
                    if Function.previous_opperation(result, i) in opp:
                        Function._append_prev_list(result, i,
                            [result[i], Function._cut_next_item(result, i)])
                        result[i] = None
                        continue
                    result[i] = [Function._cut_prev_item(result, i), result[i], 
                                 Function._cut_next_item(result, i)]

        while None in result:
            result.remove(None)
        while len(result) == 1 and isinstance(result[0], list):
            result = result[0]
        return result
    @staticmethod
    def previous_opperation(eqation: list, index):
        """Find the previous operation in the equation list."""
        sub = eqation[:index]
        for i in range(len(sub) -1, -1, -1):
            if isinstance(sub[i], Operator):
                return sub[i]
            elif isinstance(sub[i], list):
                return Function.previous_opperation(sub[i], len(sub[i]) - 1)
        return None
    @staticmethod
    def _cut_next_item(equation: list, index):
        """Cut the next item from the equation list."""
        sub = equation[index + 1:]
        for i, item in enumerate(sub):
            if item is None:
                continue
            if item != Operator.Negative:
                equation[index + i + 1] = None
                return item
            equation[index + i + 1] = None
            return [Operator.Negative, Function._cut_next_item(equation, index + i + 1)]

        print("Failed to find next item")    
    @staticmethod
    def _cut_prev_item(equation: list, index):
        """Cut the previous item from the equation list."""
        sub = equation[:index]

        for i in range(len(sub) -1, -1, -1):
            if sub[i] != None:        
                equation[i] = None
                return sub[i]
        print("Failed to find previous item")
    @staticmethod
    def _append_next_list(equation: list, index, obj):
        """Append an item to the next list in the equation list."""
        sub = equation[index + 1:]
        for i in range(len(sub)):
            if isinstance(sub[i], list):
                if isinstance(obj, list):
                    equation[index + i + 1].extend(obj)
                else:
                    equation[index + i + 1].append(obj)
                return            
        print("Failed to append next item")    
    @staticmethod
    def _append_prev_list(equation: list, index, obj):
        """Append an item to the previous list in the equation list."""
        sub = equation[:index]

        for i in range(len(sub) -1, -1, -1):
            if isinstance(sub[i], list):
                if isinstance(obj, list):
                    equation[i].extend(obj)
                else:
                    equation[i].append(obj)
                return
        print("Failed to append previous item")
    @staticmethod
    def _find_end_bracket(equation: str, index):
        """Find the index of the closing bracket in the equation string."""
        opened_brackets = 0
        sub = equation[index:]
        for i, c in enumerate(sub):
            if c == '(':
                opened_brackets += 1
            elif c == ')':
                if opened_brackets > 0:
                    opened_brackets -= 1
                    continue
                return index + i
        print("Failed to find end bracket")
    @staticmethod
    def _eq_evaluate(equation: list, variable_set, evaluate_internal=False):
        """Solve the equation using the given variable set."""
        pending_operator = None
        
        result = 0
        for item in equation:
            if isinstance(item, list):
                value = Function._eq_evaluate(item, variable_set)
            elif isinstance(item, str):
                if item in CONSTANTS:
                    value = CONSTANTS[item]
                else:
                    value = variable_set[item]
                    if isinstance(value, Function) and evaluate_internal:
                        value = value.evaluate(**variable_set)
                if value is None:
                        raise ex.UndefingedVariableError(f"Variable '{item}' is not defined")
            elif isinstance(item, (float, int)):
                value = item
            elif isinstance(item, Operator):
                pending_operator = item
                continue
            else:
                raise ex.InvalidEquationError(f"Invalid equation item: {item}")
            
            if pending_operator != None:
                if pending_operator.ismodifier():
                    result = pending_operator.preform(value)
                else:
                    result = pending_operator.preform(result, value)
                pending_operator = None
            else:
                result = value
        if isinstance(result, Function):
            result._remove_lone_parentheses()
        return result
    @staticmethod
    def _is_default_multiplier(obj):
        """Check if the object is a default multiplier."""
        return isinstance(obj, (float, int, str, list))
    @staticmethod
    def _eq_list_to_str(equation: list) -> str:
        """Convert an equation list to a string."""
        result = ""
        for i in equation:
            if isinstance(i, list):
                result += '(' + Function._eq_list_to_str(i) + ')'
                continue
            elif isinstance(i, float):
                result += str(i)
                continue
            result += str(i)
        return result

    def __call__(self, *args, **kwargs):
        """Call the Equation object as a function."""
        return self.evaluate(*args, **kwargs)    
    def __add__(self, other):
        """Add two equations."""
        new = Function(existing=self)
        if isinstance(other, Function):
            other = other.equation if len(other.equation) > 1 else other.equation[0]
        new.equation = [new.equation, Operator.Add, other]
        return new    
    def __radd__(self, other):
        """Add a scalar value to an equation."""
        new = Function(existing=self)
        if isinstance(other, Function):
            other = other.equation if len(other.equation) > 1 else other.equation[0]
        new.equation = [other, Operator.Add, new.equation]
        return new
    def __sub__(self, other):
        """Subtract two equations."""
        new = Function(existing=self)
        if isinstance(other, Function):
            other = other.equation if len(
                other.equation) > 1 else other.equation[0]
        new.equation = [new.equation, Operator.Subtract, other]
        return new    
    def __rsub__(self, other):
        """Subtract an equation from a scalar value."""
        new = Function(existing=self)
        if isinstance(other, Function):
            other = other.equation if len(
                other.equation) > 1 else other.equation[0]
        new.equation = [other, Operator.Subtract, new.equation]
        return new    
    def __mul__(self, other):
        """Multiply two equations."""
        new = Function(existing=self)
        if isinstance(other, Function):
            other = other.equation if len(
                other.equation) > 1 else other.equation[0]
        new.equation = [new.equation, Operator.Multiply, other]
        return new    
    def __rmul__(self, other):
        """Multiply a scalar value by an equation."""
        new = Function(existing=self)
        if isinstance(other, Function):
            other = other.equation if len(
                other.equation) > 1 else other.equation[0]
        new.equation = [other, Operator.Multiply, new.equation]
        return new    
    def __truediv__(self, other):
        """Divide two equations."""
        new = Function(existing=self)
        if isinstance(other, Function):
            other = other.equation if len(
                other.equation) > 1 else other.equation[0]
        new.equation = [new.equation, Operator.Divide, other]
        return new    
    def __rtruediv__(self, other):
        """Divide a scalar value by an equation."""
        new = Function(existing=self)
        if isinstance(other, Function):
            other = other.equation if len(
                other.equation) > 1 else other.equation[0]
        new.equation = [other, Operator.Divide, new.equation]
        return new    
    def __pow__(self, other):
        """Raise an equation to a power."""
        new = Function(existing=self)
        if isinstance(other, Function):
            other = other.equation if len(
                other.equation) > 1 else other.equation[0]
        new.equation = [new.equation, Operator.Power, other]
        return new    
    def __rpow__(self, other):
        """Raise a scalar value to the power of an equation."""
        new = Function(existing=self)
        if isinstance(other, Function):
            other = other.equation if len(
                other.equation) > 1 else other.equation[0]
        new.equation = [other, Operator.Power, new.equation]
        return new
    def __neg__(self):
        """Negate the equation."""
        new = Function(existing=self)
        new.equation = [Operator.Negative, new.equation]
        return new
    def __repr__(self):
        """Return the equation string."""
        return f"{Function.__name__}({Function._eq_list_to_str(self.equation)})"
    def __str__(self):
        """Return the equation string."""
        func_vars = self.function_variables()
        vars = ''.join([x + (', ' if i < len(func_vars) -1 else '') \
                        for i, x in enumerate(func_vars)])
        return f"{self.name}({vars}) = " + Function._eq_list_to_str(self.equation)