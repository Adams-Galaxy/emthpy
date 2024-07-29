"""WIP - Command line interface for emthpy"""
from console import Console, try_numeric
from _emthpy_functions_DEPRICATED import Function
from _emthpy_vectors import Vector
from _emthpy_matrices import Matrix

EVAL_CHAR = '|'
RANGE_EVAL_CHAR = '->'

def print_var(command, console):
    """Prints the value of a variable stored in the global variables dictionary"""
    if len(command.keys) > 1:
        command.pop_key()
    
    if command.active_key not in console.global_vars and command.active_key not in console.config_vars:
        print(f"Variable {command.active_key!a} not found")
        return
    var = \
        console.global_vars[command.active_key] if command.active_key in \
        console.global_vars else console.config_vars[command.active_key]
    print(var)

def store_var(command, console):
    """Stores a variable in the global variables dictionary"""
    log = command.get_parameter('log', False)

    var_name = command.pop_key()
    command.pop_key() # Remove the '='
    var_value = console.preform(command)
    console.global_vars[var_name] = var_value
    if log:
        print(f"Stored {var_name!a} as {var_value}")



mat = {
    #'inverse': inverse_mat,
    #'transpose': transpose_mat,
}
vec = {
    #'normalise': normalize_vec,
    #'magnitude': magnitude_vec,
}


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
    result = try_numeric(command.active_key)
    if isinstance(result, str):
        return Function(result)
    elif isinstance(result, (int, float)):
        return result
    else:
        raise ValueError(f"Invalid value/expression: {command.active_key!a}")
default = {
    '__global_var__': print_var,
    '__any__': _get_expression,
}

def _global(command, console):
    console.global_vars[command.pop_key()] = command.pop_key()
def _config(command, console):
    console.config_vars[command.pop_key()] = command.pop_key()
update = {
    'global' : _global,
    'config' : _config,
    '__any__': _config,
}

def num(command, console):
    command.pop_key()
    return try_numeric(command.active_key)
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
    '*percision': 2,
}


console = Console(main, "emthpy-console")
console.run()
