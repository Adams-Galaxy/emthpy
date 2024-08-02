"""WIP - Command line interface for emthpy"""
from console import Console
from _emthpy_function import Function, try_numeric
from _emthpy_vectors import Vector
from _emthpy_matrices import Matrix

EVAL_CHAR = '|'
RANGE_EVAL_CHAR = '->'


def get_expression_params(string, existing_vars=None):
    """Returns a tuple containing the arguments and keyword arguments from a string"""
    if string == '':
        return (), {}

    args, kwargs = [], existing_vars.copy() if existing_vars else {}

    for item in string.split(','):
        if '=' in item:
            key, value = item.split('=')
            kwargs[key] = try_numeric(value)
        else:
            args.append(try_numeric(item, allow_expression=True))
    result = tuple(args), kwargs
    return result
def print_var(command, console):
    """Prints the value of a variable stored in the global variables dictionary"""
    log = command.get_parameter('log', True)

    # Check if the print_var is called via 'print' or
    # default (e.g inputting only a variable name)
    if command.active_key == 'print':
        command.pop_key()
    
    # Check if the variable is in the global variables dictionary
    if command.active_key not in console.global_vars and \
        command.active_key not in console.config_vars:
        print(f"Variable {command.active_key!a} not found")
        print(f"Global variables: {console.global_vars}")
        print(f"Config variables: {console.config_vars}")
        return
    var = \
        console.global_vars[command.active_key] if command.active_key in \
        console.global_vars else console.config_vars[command.active_key]

    # Check if the variable is a function
    if log:
        if isinstance(var, Function):
            print(var.notation_str()) # Print the function in notation form
        else:
            print(var)
    return var
def store_var(command, console):
    """Stores a variable in the global variables dictionary"""
    log = command.get_parameter('log', False)
    command.set_parameter('log', False)

    var_name = command.pop_key()
    command.pop_key() # Remove the '='
    var_value = console.preform(command) # Get the value of the variable

    # Check if the variable is a function, if so, set the name of the function
    if isinstance(var_value, Function):
        if var_name in var_value.variables():
            raise ValueError(f"Variable {var_name!a} is in the equation {var_value!a}")
        var_value.name = var_name

    console.global_vars[var_name] = var_value
    if log:
        print(f"Stored {var_name!a} as {var_value}")

# Logic for matrix operations
def _inverse_mat(command, console):
    log = command.get_parameter('log', True)
    command.set_parameter('log', False)

    command.pop_key() # Skip the 'inverse' key
    matrix = console.preform(command)
    if not isinstance(matrix, Matrix):
        raise ValueError(f"Invalid input: {matrix}, expected Matrix")
    if not matrix.inverse():
        print("Matrix is not invertible")
        return
def _inversed_mat(command, console):
    log = command.get_parameter('log', True)
    command.set_parameter('log', False)

    command.pop_key() # Skip the 'inversed' key
    matrix = console.preform(command)
    if not isinstance(matrix, Matrix):
        raise ValueError(f"Invalid input: {matrix}, expected Matrix")
    matrix = matrix.copy()
    if not matrix.inverse():
        print("Matrix is not invertible")
        return
    if log:
        print(matrix)
    return
def _transpose_mat(command, console):
    log = command.get_parameter('log', True)
    command.set_parameter('log', False)

    command.pop_key() # Skip the 'transpose' key
    matrix = console.preform(command)
    if not isinstance(matrix, Matrix):
        raise ValueError(f"Invalid input: {matrix}, expected Matrix")
    matrix.transpose()
    return
def _transposed_mat(command, console):
    log = command.get_parameter('log', True)
    command.set_parameter('log', False)

    command.pop_key() # Skip the 'transposed' key
    matrix = console.preform(command)
    if not isinstance(matrix, Matrix):
        raise ValueError(f"Invalid input: {matrix}, expected Matrix")
    matrix = matrix.copy()
    matrix.transpose()
    if log:
        print(matrix)
    return
def _determinant_mat(command, console):
    log = command.get_parameter('log', True)
    command.set_parameter('log', False)

    command.pop_key() # Skip the 'determinant' key
    matrix = console.preform(command)
    if not isinstance(matrix, Matrix):
        raise ValueError(f"Invalid input: {matrix}, expected Matrix")
    result = matrix.determinant
    if log:
        print(result)
    return result
mat = {
    'inverse': _inverse_mat,
    'inversed': _inversed_mat,
    'transpose': _transpose_mat,
    'transposed': _transposed_mat,
    'determinant': _determinant_mat,
    'det': _determinant_mat,
}

# Logic for vector operations
def _normalize_vec(command, console):
    log = command.get_parameter('log', True)
    command.set_parameter('log', False)

    command.pop_key() # Skip the 'normalise' key
    vector = console.preform(command)
    if not isinstance(vector, Vector):
        raise ValueError(f"Invalid input: {vector}, expected Vector")
    vector.normalize()
    return
def _normalized_vec(command, console):
    log = command.get_parameter('log', True)
    command.set_parameter('log', False)

    command.pop_key() # Skip the 'normalized' key
    vector = console.preform(command)
    if not isinstance(vector, Vector):
        raise ValueError(f"Invalid input: {vector}, expected Vector")
    vector = vector.copy()
    vector.normalize()
    if log:
        print(vector)
    return
def _magnitude_vec(command, console):
    log = command.get_parameter('log', True)
    command.set_parameter('log', False)

    command.pop_key() # Skip the 'magnitude' key
    vector = console.preform(command)
    if not isinstance(vector, Vector):
        raise ValueError(f"Invalid input: {vector}, expected Vector")
    result = vector.magnitude
    if log:
        print(result)
    return result
vec = {
    'normalise': _normalize_vec,
    'normalized': _normalized_vec,
    'magnitude': _magnitude_vec,
}

# Logic for creating new objects
def _new_vec(command, console):
    return Vector([try_numeric(x) for x in input().split()])
def _new_mat(command, console):
    result = []
    inpt = input()
    while inpt != '':
        result.append(inpt.split())
        inpt = input()

    for row in result:
        for i, item in enumerate(row):
            num_item = try_numeric(item)
            if not isinstance(num_item, (int, float)):
                raise ValueError(f"Invalid input: {item}")
            row[i] = num_item
    return Matrix(result)
new = {
    'vec': _new_vec,
    'mat': _new_mat,
}


def _get_expression(command, console):
    """
    Returns a Function object from the command key
    
    TODO: 
        - Ensure the function cannot loop infinitely (e.g 2x|x=a,a=x)
    """
    log = command.get_parameter('log', True)

    func, params = command.active_key, ()
    if EVAL_CHAR not in command.active_key: # Check if there are evaluation parameters
        if log:
            print(func)
        return Function(func)

    func, params = command.active_key.split(EVAL_CHAR)
    vars, tmp = get_expression_params(params) # Get the arguments and keyword arguments (in *vars, **kwvars form)
    kwvars = console.global_vars.copy()
    kwvars.update(tmp) # Copy the global variables and update them with the evaluation parameters

    # Check if the function (or rather its name e.g 'f') is in the global variables
    if func in console.global_vars and isinstance(console.global_vars[func], Function):
        func = console.global_vars[func]
        result = func(*vars, **kwvars)
    else:
        # assign the function to a Function object
        result = Function(func)(*vars, **kwvars)

    # Evaluate the function until it is no longer
    # a function, or cannot be evaluated further
    while isinstance(result, Function) and \
        result.satisfied(**kwvars):
        result = result.evaluate(**kwvars)
    if log:
        print(result)
    return result
default = {
    '__config_var__': print_var,
    '__global_var__': print_var,
    '__any__': _get_expression,
}

def _global(command, console):
    key = command.pop_key()
    console.global_vars[key] = try_numeric(command.pop_key())
def _config(command, console):
    key = command.pop_key()
    console.config_vars[key] = try_numeric(command.pop_key())
update = {
    'global' : _global,
    'config' : _config,
    '__any__': _config,
}

main = {
    'print': print_var,
    'update': update,
    'mat': mat,
    'vec': vec,
    'new': new,
    'quit': lambda command, console: console.stop(),
    'q': lambda command, console: console.stop(),
    '__assingment__': store_var,
    '__any__': default,
}

config = {
    'dp': 2,
}

console = Console(main, "emthpy-console")
console.global_vars.update({
    'i' : Vector([1, 0, 0]),
    'j' : Vector([0, 1, 0]),
    'k' : Vector([0, 0, 1]),
    'A' : Matrix([
        [1, 2, 3],
        [0, 1, 4],
        [5, 6, 0],
        ]),
    'B' : Matrix([
        [1, 2, 3],
        [0, 1, 4],
        [5, 6, 0],
        ]),
})
console.run()
