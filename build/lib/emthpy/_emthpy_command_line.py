from _command_line import CommandSet, numeric
from _emthpy_functions import CONSTANTS, Function
from _emthpy_matrices import Matrix
from _emthpy_vectors import Vector

EVAL_CHAR = '|'
RANGE_EVAL_CHAR = '->'

def format_numeric(x, global_vars):
    if isinstance(x, float):
        return f"{x:.{global_vars['*percision']}f}"
    return x

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

def defign_var(name, *args, **kwargs):
    global_vars = kwargs.get('global_vars')
    if isinstance(args[1], (int, float)):
        global_vars['inst'].add_global_var(
            name, args[1])
    else:
        value = global_vars['inst'].run_command(split_command=list(args[1:]), **kwargs)
        if isinstance(value, Function):
            if value.contains_any(name):
                raise ValueError(f"Variable {name!a} is in the equation")
            value.name = name
        global_vars['inst'].add_global_var(name, value)

def new_vec(*args, **kwargs):
    global_vars = kwargs.get('global_vars')
    return Vector([numeric(x) for x in input().split()])

def new_mat(*args, **kwargs):
    global_vars = kwargs.get('global_vars')
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

def print_var(var_name, **kwargs):
    global_vars = kwargs.get('global_vars')
    """
    Print the value of a variable.

    Args:
        var_name: The name of the variable.
        kwargs: Additional keyword arguments.

    Returns:
        None
    """
    global_vars['inst'].output(global_vars[var_name], **kwargs)

def run_expression(eq_inpt, **kwargs):
    global_vars = kwargs.get('global_vars')
    log = kwargs.get('log', False)
    log_on_eval = kwargs.get('log-on-eval', False)

    if isinstance(eq_inpt, (int, float)):
        return eq_inpt

    if EVAL_CHAR in eq_inpt:
        eq_inpt, vars = eq_inpt.split('|')
        log = log_on_eval
    else:
        eq_inpt, vars = eq_inpt, ''

    if eq_inpt in global_vars and isinstance(global_vars[eq_inpt], Function):
        eq = global_vars[eq_inpt]
    else:
        eq = Function(eq_inpt, **global_vars, **kwargs)
        while isinstance(eq, Function) and eq.vars_satisfied():
            eq = eq()

    if RANGE_EVAL_CHAR in vars:
        a, b = vars.split(RANGE_EVAL_CHAR)
        a, b = eval_args_from_str(a, existing_vars=global_vars), eval_args_from_str(
            b, existing_vars=global_vars)
        result = eq.evaluate_from(a, b)

        while isinstance(result, Function) and result.vars_satisfied(**global_vars):
            result = result.evaluate(**global_vars)
        
        if log:
            global_vars['inst'].output(result, **kwargs)
        return result

    var_args, var_kwargs = eval_args_from_str(vars)
    var_kwargs.update(global_vars)

    while isinstance(eq, Function) and eq.vars_satisfied(*var_args, **var_kwargs):
        eq = eq.evaluate(*var_args, **var_kwargs)
    
    if log:
        global_vars['inst'].output(eq, **kwargs)
    return eq

def inverse_mat(*args, **kwargs):
    global_vars = kwargs.get('global_vars')
    if len(args) > 0 and args[0] in global_vars:
        result = global_vars[args[0]].copy()
    else:
        result = new_mat()

    if result.inverse():
        global_vars['inst'].output(result, **kwargs)
        return result
    global_vars['inst'].output("Matrix is not invertible", **kwargs)
    return None

def transpose_mat(*args, **kwargs):
    global_vars = kwargs.get('global_vars')
    if len(args) > 0 and args[0] in global_vars:
        result = global_vars[args[0]]
    else:
        result = new_mat()

    result = result.T
    global_vars['inst'].output(result, **kwargs)
    return result

def normalize_vec(*args, **kwargs):
    global_vars = kwargs.get('global_vars')
    if len(args) > 0 and args[0] in global_vars:
        result = global_vars[args[0]]
    else:
        result = new_vec()

    result = result.normalised()
    global_vars['inst'].output(result, **kwargs)
    return result

def magnitude_vec(*args, **kwargs):
    global_vars = kwargs.get('global_vars')
    if len(args) > 0 and args[0] in global_vars:
        result = global_vars[args[0]]
    else:
        result = new_vec()

    result = result.magnitude
    global_vars['inst'].output(result, **kwargs)
    return result

def quit(**kwargs):
    """
    Quit the program.

    Args:
        kwargs: Additional keyword arguments.

    Returns:
        None
    """
    exit()

def trap(subject, start, stop, n=-1, a=10, *args, **kwargs):
    """
    Calculate the trapezoidal integral of a subject.

    Args:
        subject: The subject to integrate.
        args: Additional arguments.
        kwargs: Additional keyword arguments.

    Returns:
        None
    """
    global_vars = kwargs.get('global_vars')
    # global_vars['inst'].output(subject, kwargs)
    if subject in global_vars:
        result = global_vars[subject].trap_intergral(start, stop, n, a)
        global_vars['inst'].output(result, "u^2", **kwargs)
        return result
    subject = Function(subject)
    result = subject.trap_intergral(start, stop, n, a)
    global_vars['inst'].output(result, "u^2", **kwargs)
    return result

def simps(subject, start, stop, n=-1, a=10, *args, **kwargs):
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
    global_vars = kwargs.get('global_vars')
    if subject in global_vars:
        result = global_vars[subject].simps_intergral(start, stop, n, a)
        global_vars['inst'].output(
            result, "u^2", **kwargs)
        return result
    subject = Function(subject)
    result = subject.simps_intergral(start, stop, n, a)
    global_vars['inst'].output(result, "u^2", **kwargs)
    return result

def limit(x, *args, **kwargs):
    """
    Calculate the limit of a subject.

    Args:
        subject: The subject to calculate the limit of.
        x: The variable to calculate the limit for.
        a: The value of the variable.
        args: Additional arguments.
        global_vars: Global variables dictionary.
        kwargs: Additional keyword arguments.

    Returns:
        The result of the limit.

    Raises:
        ValueError: If the input is invalid.
    """
    global_vars = kwargs.get('global_vars')
    subject = global_vars['inst'].run_command(split_command=list(args), **kwargs)
    if isinstance(subject, (int, float)):
        global_vars['inst'].output(subject, **kwargs)
        return subject
    x = numeric(x.split('->')[1])
    if subject in global_vars:
        result = global_vars[subject].limit(x)
        global_vars['inst'].output(result, **kwargs)
        return result
    if isinstance(subject, Function):
        result = subject.limit(x)
        global_vars['inst'].output(result, **kwargs)
        return result
    raise ValueError(f"Invalid input: {subject}")

def print_vars(*args, **kwargs):
    global_vars = kwargs.get('global_vars')
    for key, value in global_vars.items():
        global_vars['inst'].output(f"{key}: {repr(value)}", **kwargs)

def print_args(*args, **kwargs):
    global_vars = kwargs.get('global_vars')
    global_vars['inst'].output(args, kwargs, **kwargs)



mat = {
    'inverse': inverse_mat,
    'transpose': transpose_mat,
}

vec = {
    'normalise': normalize_vec,
    'magnitude': magnitude_vec,
}

eq = {

}

new = {
    'vec': new_vec,
    'mat': new_mat,
}

default = {
    '__global-var__': print_var,
    '__any__': (run_expression, {'log-on-eval': True}),
    }

main = {
    'print': print_var,
    'vars': print_vars,
    'args': print_args,
    'mat': mat,
    'vec': vec,
    'eq': eq,
    'new': new,
    'trap': trap,
    'simps': simps,
    'limit': limit,
    'lim': limit,
    'run': run_expression,
    'quit': quit,
    'q': quit,
    '__assingment__': defign_var,
    '__any__': default,
}

config = {
    '*percision': 2,
}

cmd = CommandSet(main, pre_prompt="emthpy-console: ", **config)


def run_command_line(debug=False):
    while True:
        try:
            cmd.run_command()
        except Exception as ex:
            if debug:
                raise ex
            cmd.output("Invalid syntax, error: " + str(ex))


if __name__ == '__main__':
    run_command_line(True)
