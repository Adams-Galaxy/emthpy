"""WIP - Command line interface for emthpy"""

from _emthpy_types import Function
from _emthpy_functions_DEPRICATED import CONSTANTS
from sympy.core.numbers import Float as spFloat
from sympy.core import Rational

PARAM_CHAR = '@'
OPTION_CHAR = '--'
BOOLS = ('==', '!=', '>', '<', '>=', '<=')


def numeric(x, thow_error=True):
    """Converts a string to a number if it is a number."""
    def convert(s):
        if isinstance(s, (int, float)):
            return s
        if s in CONSTANTS:
            return CONSTANTS[s]
        formated = s.replace('.', '').replace('-', '')
        if not formated.isnumeric():
            if thow_error:
                raise ValueError(f"Invalid number: {s}")
            return s
        return float(s) if '.' in s else int(s)
    
    if isinstance(x, (list, tuple)):
        result = []
        for item in x:
            result.append(convert(item))
        return tuple(result)
    return convert(x)


DUNNER_COMMANDS = {
    '__bool__': lambda *args, **kwargs: args[1] in BOOLS if len(args) > 1 else False,
    '__assingment__': lambda *args, **kwargs: args[1] == '=' if len(args) > 1 else False,
    '__global-var__': lambda *args, **kwargs: args[0] in kwargs['global_vars'],
    '__any__': lambda *args, **kwargs: True,
    }

class CommandSet:
    def __init__(self, command_dict, pre_prompt="", **kwargs):
        self.pre_prompt = pre_prompt
        self.command_dict = command_dict
        self.global_vars = kwargs
        self.global_vars['inst'] = self

    @staticmethod
    def extract_kwargs(prompt):
        """Returns a dictionary of kwargs from a list of prompt items."""
        to_remove = []
        kwargs = {}
        for i, item in enumerate(prompt):
            if not isinstance(item, str):
                continue
            if item.startswith(OPTION_CHAR):
                key, value = item[len(OPTION_CHAR):], True
                kwargs[key] = value
                to_remove.append(item)
            elif item.startswith(PARAM_CHAR):
                key, value = item[len(PARAM_CHAR):], prompt[i+1]
                kwargs[key] = value
                to_remove.append(item)
                to_remove.append(prompt[i+1])
        for item in to_remove:
            prompt.remove(item)
        return kwargs
    
    def add_global_var(self, var_name, var):
        self.global_vars[var_name] = var

    def run_command(self, command=None, split_command=None, **kwargs):
        def evaluate_part(parts, commands_dict):
            part = parts[0]
            args_start = 1
            if part not in commands_dict:
                for dunner_command, func in DUNNER_COMMANDS.items():
                    if dunner_command in commands_dict and func(*parts, **kwargs):
                        part = dunner_command
                        break
                else:
                    raise ValueError(f"{part} not in target:{commands_dict}")
                args_start = 0
            result = commands_dict[part]

            if isinstance(result, tuple):
                result, add_kwargs = result
                kwargs.update(add_kwargs)

            if callable(result):
                return result(*parts[args_start:], **kwargs)
            elif isinstance(result, dict):
                return evaluate_part(parts[args_start:], result)
            else:
                raise ValueError(
                    f"Invalid target type: {type(result).__name__}")
        
        if split_command is None:
            if command is None:
                command = input(self.pre_prompt)
            if command == '':
                return None
            command = command.split()
            for i, item in enumerate(command):
                command[i] = numeric(item, False)
        else:
            command = split_command

        kwargs.update(CommandSet.extract_kwargs(command))
        kwargs['global_vars'] = self.global_vars
        return evaluate_part(command, self.command_dict)

    def output(self, *args, **kwargs):
        round_floats = not kwargs.get('disable-rounding', False)
        dp = kwargs.get('dp', self.global_vars['*percision'])
        def get_str_value(value):
            if isinstance(value, Rational):
                value = float(value)
            if round_floats and isinstance(value, (float, spFloat)):
                return format(value, f".{dp}f")
            return str(value)
        
        result = []
        for item in args:
            result.append(get_str_value(item))
        print(' '.join(result))

    def __repr__(self) -> str:
        return f"{CommandSet.__name__}(pre_prompt={self.pre_prompt!a})"

