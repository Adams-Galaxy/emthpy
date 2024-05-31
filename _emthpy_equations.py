import math
from enum import Enum
import inspect
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

    def placeholder(self):
        """Get the placeholder string for the operator."""
        return "&" + self.value + "&"
    
    def ismodifier(self):
        """Check if the operator is a modifier (takes one argument)."""
        return len(inspect.signature(OPERATIONS[self]).parameters) == 1

    def preform(self, *args):
        """Perform the operation with the given arguments."""
        return OPERATIONS[self](*args)

    def __repr__(self):
        """Return the string representation of the operator."""
        return f"{Operator.__name__}.{self.name}: \'{self.value}\'"

    def __str__(self):
        """Return the string representation of the operator."""
        return self.value

# Define a dictionary of mathematical constants
CONSTANTS = {'e': math.e, 'pi': math.pi}

# Define the order of operations (BEDMAS)
BEDMAS = (
    (Operator.Negative,),
    (Operator.Sin, Operator.Cos, Operator.Tan, Operator.ASin, Operator.ACos, Operator.ATan,
     Operator.Sec, Operator.Cosec, Operator.Cot, Operator.LogX, Operator.Log, Operator.Ln),
    (Operator.Power, Operator.Root),
    (Operator.Multiply, Operator.Divide),
    (Operator.Add, Operator.Subtract)
)

# Define a dictionary of operator functions
OPERATIONS = {
    Operator.Sin: math.sin,
    Operator.Cos: math.cos,
    Operator.Tan: math.tan,
    Operator.ASin: math.asin,
    Operator.ACos: math.acos,
    Operator.ATan: math.atan,
    Operator.Sec: lambda x: 1/math.cos(x),
    Operator.Cosec: lambda x: 1/math.sin(x),
    Operator.Cot: lambda x: 1/math.tan(x),
    Operator.LogX: lambda x, y: math.log(x, y),
    Operator.Log: math.log10,
    Operator.Ln: lambda x: math.log(x, math.e),
    Operator.Power: pow,
    Operator.Root: math.sqrt,
    Operator.Divide: lambda x, y: x / y,
    Operator.Multiply: lambda x, y: x * y,
    Operator.Add: lambda x, y: x + y,
    Operator.Subtract: lambda x, y: x - y,
    Operator.Negative: lambda x: -x
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
    Operator.Power: ['**', '^'],
    Operator.Root: 'sqrt',
    Operator.Add: '+',
    Operator.Subtract: '-',
    Operator.Multiply: '*',
    Operator.Divide: '/'
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


class Equation:
    """Class representing a mathematical equation."""
    def __init__(self, equation=..., *args, variable_set=..., existing=..., require_format=True, **kwargs):
        if existing != ...:
            self.init_from_existing(existing, *args, **kwargs)
            return

        # Initialize the Equation object
        if variable_set != ...:
            self.variable_set = variable_set
        else:
            self.variable_set = VariableSet()

        if isinstance(equation, str):
            self.init_from_str(equation, *args, **kwargs)
        elif isinstance(equation, list):
            self.equation = equation
            if require_format:
                self.equation = Equation.format(equation)
        self.update_variables(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        """Create an Equation object."""
        return super().__new__(cls)

    def init_from_str(self, equation, *args, **kwargs):
        """Initialize the Equation object from a string."""
        
        # Refactor the equation string to replace functions with single char identifiers
        equation = self.refactor_str_operators(equation + '\r')

        # Interpret the refactored equation string
        self.unformated = self.interperate(equation)

        # Format the interpreted equation
        self.equation = Equation.format(self.unformated)

        self.update_variables(*args, **kwargs)

    def init_from_existing(self, equation, *args, **kwargs):
        """Initialize the Equation object from an existing Equation object."""
        self.equation = equation.equation[:]
        self.unformated = equation.unformated[:]
        self.variable_set = equation.variable_set
        self.update_variables(*args, **kwargs)

    @classmethod
    def from_numeric(cls, value, *args, **kwargs):
        """Create an Equation object from a numeric value."""
        return cls(existing=[value], *args, **kwargs)

    def decomposed(self):
        """Decompose the equation into its components."""

        result = []
        for item in self.equation:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result

    def requires_format(self):
        """Check if the equation requires formatting."""
        return True in [isinstance(item, list) for item in self.equation]

    @classmethod
    def from_existing(cls, equation, *args, **kwargs):
        """Create an Equation object from an existing Equation object."""
        return cls(existing=equation, *args, **kwargs)

    def update_variables(self, *args, **kwargs):
        """Update the variable values."""
        # Assign values to variables if provided as arguments
        if len(args) > 0:
            if len(args) != len(self.variable_set):
                print(args)
                raise ex.InvalidArgumentError("Invalid number of arguments (" +
                                           f"{len(args)}) for {len(self.variable_set)} variables")
            for i, var in enumerate(self.variable_set.keys()):
                self.variable_set[var] = args[i]

        # Assign values to variables if provided as keyword arguments
        for key, value in kwargs.items():
            self.variable_set[key] = value

    def evaluate(self, *args, **kwargs):
        """Solve the equation with the given variable values."""
        # Update the variables if arguments are provided
        self.update_variables(*args, **kwargs)
        return Equation.solve_equation(self.equation, self.variable_set)
         
    def refactor_str_operators(self, equation: str) -> str:
        """Replace functions in the equation string with single char identifiers."""
        # Replace functions (e.g sin) with single char identifiers
        for key, value in OPERATOR_KEYS.items():
            if isinstance(value, list):
                for s in value:
                    equation = equation.replace(s, key.placeholder())
                continue
            equation = equation.replace(value, key.placeholder())

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
                if Equation.is_default_multiplier(last_result):
                    if not pending_operator:
                        result.append(Operator.Multiply)
                alpha_values = Equation.find_var_in_str(equation[i:])
                result.append(alpha_values)

                self.variable_set[alpha_values] = None
                pending_operator = False
                skip_to = i + len(alpha_values)
            elif c == '(':
                endpos = Equation.find_end_bracket(equation, i+1)
                substring = equation[i+1:endpos]
                skip_to = endpos + 1
                # Check for implied multiplication
                if Equation.is_default_multiplier(last_result):
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
                if Equation.is_default_multiplier(last_result):
                    if not pending_operator:
                        result.append(Operator.Multiply)
                result.append(substring)
            else:
                raise ex.UnkownOperatorError(f"Unkown operator: \'{c}\'")
        return result
    
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
                result[i] = Equation.format(item)
        
        for opp in BEDMAS:
            for i, item in enumerate(result):
                if item in opp:
                    if item.ismodifier():
                        result[i] = [item, Equation.cut_next_item(result, i)]
                        continue
                    if Equation.previous_opperation(result, i) in opp:
                        Equation.append_prev_list(result, i,
                            [result[i], Equation.cut_next_item(result, i)])
                        result[i] = None
                        continue
                    result[i] = [Equation.cut_prev_item(result, i), result[i], 
                                 Equation.cut_next_item(result, i)]

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
                return Equation.previous_opperation(sub[i], len(sub[i]) - 1)
        return None

    @staticmethod
    def cut_next_item(equation: list, index):
        """Cut the next item from the equation list."""
        sub = equation[index + 1:]
        for i, item in enumerate(sub):
            if item is None:
                continue
            if item != Operator.Negative:
                equation[index + i + 1] = None
                return item
            equation[index + i + 1] = None
            return [Operator.Negative, Equation.cut_next_item(equation, index + i + 1)]

        print("Failed to find next item")
    
    @staticmethod
    def cut_prev_item(equation: list, index):
        """Cut the previous item from the equation list."""
        sub = equation[:index]

        for i in range(len(sub) -1, -1, -1):
            if sub[i] != None:        
                equation[i] = None
                return sub[i]
        print("Failed to find previous item")

    @staticmethod
    def append_next_list(equation: list, index, obj):
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
    def append_prev_list(equation: list, index, obj):
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
    def find_end_bracket(equation: str, index):
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
    def solve_equation(equation: list, variable_set):
        """Solve the equation using the given variable set."""
        pending_operator = None
        
        result = 0.0
        for item in equation:
            if isinstance(item, list):
                value = Equation.solve_equation(item, variable_set)
            elif isinstance(item, str):
                if item in CONSTANTS:
                    value = CONSTANTS[item]
                else:
                    value = variable_set[item]
            elif isinstance(item, (float, int)):
                value = item
            elif isinstance(item, Operator):
                pending_operator = item
                continue
            
            if pending_operator != None:
                if pending_operator.ismodifier():
                    result = pending_operator.preform(value)
                else:
                    result = pending_operator.preform(result, value)
                pending_operator = None
            else:
                result = value
        return result
    
    @staticmethod
    def is_default_multiplier(obj):
        """Check if the object is a default multiplier."""
        return isinstance(obj, (float, int, str, list))

    @staticmethod
    def eq_list_to_str(equation: list) -> str:
        """Convert an equation list to a string."""
        result = ""
        for i in equation:
            if isinstance(i, list):
                result += '(' + Equation.eq_list_to_str(i) + ')'
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
        new = Equation(existing=self)
        new.equation = [new.equation, Operator.Add, other.equation if isinstance(other, Equation) else other]
        return new
    
    def __radd__(self, other):
        """Add a scalar value to an equation."""
        new = Equation(existing=self)
        new.equation = [other.equation if isinstance(other, Equation) else other,
                        Operator.Add, new.equation]
        return new

    def __sub__(self, other):
        """Subtract two equations."""
        new = Equation(existing=self)
        new.equation = [new.equation, Operator.Subtract, other.equation \
                        if isinstance(other, Equation) else other]
        return new
    
    def __rsub__(self, other):
        """Subtract an equation from a scalar value."""
        new = Equation(existing=self)
        new.equation = [other.equation if isinstance(other, Equation) else other,
                        Operator.Subtract, new.equation]
        return new
    
    def __mul__(self, other):
        """Multiply two equations."""
        new = Equation(existing=self)
        new.equation = [new.equation, Operator.Multiply, other.equation \
                        if isinstance(other, Equation) else other]
        return new
    
    def __rmul__(self, other):
        """Multiply a scalar value by an equation."""
        new = Equation(existing=self)
        new.equation = [other.equation if isinstance(other, Equation) else other,
                        Operator.Multiply, new.equation]
        return new
    
    def __truediv__(self, other):
        """Divide two equations."""
        new = Equation(existing=self)
        new.equation = [new.equation, Operator.Divide, other.equation \
                        if isinstance(other, Equation) else other]
        return new
    
    def __rtruediv__(self, other):
        """Divide a scalar value by an equation."""
        new = Equation(existing=self)
        new.equation = [other.equation if isinstance(other, Equation) else other,
                        Operator.Divide, new.equation]
        return new
    
    def __pow__(self, other):
        """Raise an equation to a power."""
        new = Equation(existing=self)
        new.equation = [new.equation, Operator.Power, other.equation \
                        if isinstance(other, Equation) else other]
        return new
    
    def __rpow__(self, other):
        """Raise a scalar value to the power of an equation."""
        new = Equation(existing=self)
        new.equation = [other.equation if isinstance(other, Equation) else other,
                        Operator.Power, new.equation]
        return new

    def __repr__(self):
        """Return the equation string."""
        return Equation.eq_list_to_str(self.equation)

    def __str__(self):
        """Return the equation string."""
        return Equation.eq_list_to_str(self.equation)