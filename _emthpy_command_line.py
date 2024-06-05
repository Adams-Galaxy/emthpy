import _command_line as cl
from _emthpy_equations import CONSTANTS, Equation
from _emthpy_matrices import Matrix
from _emthpy_vectors import Vector

def numeric(x):
    """
    Convert a value to a numeric type.

    Args:
        x: The value to convert.

    Returns:
        The converted numeric value.
    """
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, (tuple, list)):
        result = list(x)
        for i, item in enumerate(result):
            if isinstance(item, str):
                result[i] = float(item) if '.' in item else int(item)
        return tuple(result)
    if x in CONSTANTS:
        return CONSTANTS[x]
    if not x.replace('.','').isnumeric():
        return x
    return float(x) if '.' in x else int(x)
def eval_args_from_str(string, existing_vars={}):
    if string == '':
        return (), {}

    args, kwargs = [], existing_vars.copy()

    for item in string.split(','):
        if '=' in item:
            key, value = item.split('=')
            kwargs[key] = numeric(value)
        else:
            args.append(numeric(item))
    result = tuple(args), kwargs
    return result


def defign_var(name, bool, *args, global_vars=..., **kwargs):
    if isinstance(args[0], (int, float)):
        global_vars['inst'].add_global_var(
            name, args[0])
    else:
        value = global_vars['inst'].run_command(split_command=list(args))
        if isinstance(value, Equation) and value.contains_any(name):
            raise ValueError(f"Variable {name!a} is in the equation")
        global_vars['inst'].add_global_var(name, value)
def new_vec(*args, global_vars=..., **kwargs):
    return Vector([numeric(x) for x in input().split()])
def new_mat(*args, global_vars=..., **kwargs):
    matrix = []
    inpt = input()
    while inpt != '':
        matrix.append(inpt.split())
        inpt = input()

    for row in matrix:
        for i, item in enumerate(row):
            if item.isnumeric():
                row[i] = float(item) if '.' in item else int(item)
    return Matrix(matrix)
def print_var(var_name, global_vars=..., **kwargs):
    """
    Print the value of a variable.

    Args:
        var_name: The name of the variable.
        kwargs: Additional keyword arguments.

    Returns:
        None
    """
    print(global_vars[var_name])
def run_eq(eq_str, global_vars=..., **kwargs):
    """
    Run an equation.

    Args:
        eq_str: The equation string.
        kwargs: Additional keyword arguments.

    Returns:
        None

    Raises:
        ValueError: If the input is invalid.
    """

    if '|' not in eq_str:
        eq_str, vars = eq_str, ''
    else:   eq_str, vars = eq_str.split('|')

    if eq_str in global_vars and isinstance(global_vars[eq_str], Equation):
        eq = global_vars[eq_str]
    else:
        eq = Equation(eq_str, **global_vars)

    if '-' in vars:
        a, b = vars.split('-')
        a, b = eval_args_from_str(a, existing_vars=global_vars), eval_args_from_str(
            b, existing_vars=global_vars)
        result = eq.evaluate_from(a, b)
        
        while isinstance(result, Equation) and result.vars_satisfied(**global_vars):
            result = result.evaluate(**global_vars)
        print(result)
        return result
    
    args, new_kwargs = eval_args_from_str(vars)
    global_vars.update(new_kwargs)

    if not eq.vars_satisfied(*args, **new_kwargs):
        print(eq)
        return eq
    
    result = eq.evaluate(*args, **global_vars)
    while isinstance(result, Equation) and result.vars_satisfied(**global_vars):
        result = result.evaluate(**global_vars)
    print(result)
    return result
def inverse_mat(*args, global_vars=..., **kwargs):
    if len(args) > 0 and args[0] in global_vars:
        result = global_vars[args[0]].copy()
    else:
        result = new_mat()

    if result.inverse():
        print(result)
        return result
    print("Matrix is not invertible")
    return None
def quit(global_vars=..., **kwargs):
    """
    Quit the program.

    Args:
        kwargs: Additional keyword arguments.

    Returns:
        None
    """
    exit()
def trap(subject, start, stop, n=-1, a=10, *args, global_vars=..., **kwargs):
    """
    Calculate the trapezoidal integral of a subject.

    Args:
        subject: The subject to integrate.
        args: Additional arguments.
        kwargs: Additional keyword arguments.

    Returns:
        None
    """
    #print(subject, kwargs)
    if subject in global_vars:
        result = global_vars[subject].trap_intergral(start, stop, n, a)
        print(f"{result:.{global_vars['*percision']}f} u^2")
        return result
    subject = Equation(subject)
    result = subject.trap_intergral(start, stop, n, a)
    print(f"{result:.{global_vars['*percision']}f} u^2")
    return result
def simps(subject, start, stop, n=-1, a=10, *args, global_vars=..., **kwargs):
    """
    Calculate the Simpson's rule integral of a subject.

    Args:
        subject: The subject to integrate.
        start: The starting point of integration.
        stop: The stopping point of integration.
        n: The number of intervals for integration. Default is -1.
        a: The coefficient for the Simpson's rule. Default is 10.
        args: Additional arguments.
        global_vars: Global variables dictionary.
        kwargs: Additional keyword arguments.

    Returns:
        The result of the Simpson's rule integral.

    Raises:
        ValueError: If the input is invalid.
    """
    if subject in global_vars:
        result = global_vars[subject].simps_intergral(start, stop, n, a)
        print(f"{result:.{global_vars['*percision']}f} u^2")
        return result
    subject = Equation(subject)
    result = subject.simps_intergral(start, stop, n, a)
    print(f"{result:.{global_vars['*percision']}f} u^2")
    return result
def print_vars(*args, global_vars=..., **kwargs):
    for key, value in global_vars.items():
        print(f"{key}: {value}")
def print_args(*args, global_vars=..., **kwargs):
    print(args, kwargs)

mat = {
    'inverse': inverse_mat,
}
vec = {

}
eq = {

}
new = {
    'vec': new_vec,
    'mat': new_mat,
}
main = {
    'print' : print_var,
    'vars': print_vars,
    'args': print_args,
    'mat': mat,
    'vec': vec,
    'eq': eq,
    'new': new,
    'trap': trap,
    'run': run_eq,
    'quit': quit,
    'q': quit,
    '__assingment__': defign_var,
    '__any__': run_eq,
}
config = {
    '*percision': 2,
}

cmd = cl.CommandSet(main, pre_prompt="emthpy-console: ", **config)
def run_command_line(debug=False):
    while True:
        try:
            cmd.run_command()
        except Exception as ex:
            if debug:
                raise ex
            print("Invalid syntax, error: ", ex)

if __name__ == '__main__':
    run_command_line()