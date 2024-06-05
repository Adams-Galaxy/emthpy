from _emthpy_types import Equation

def numeric(x):
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, tuple):
        result = list(x)
        for i, item in enumerate(result):
            if isinstance(item, str):
                result[i] = float(item) if '.' in item else int(item)
        return tuple(result)
    return float(x) if '.' in x else int(x)

DUNNER_COMMANDS = {
    '__assingment__': lambda *args, **kwargs: args[1] == '=' if len(args) > 1 else False,
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
            if item.startswith('--'):
                key, value = item[2:], True
                kwargs[key] = value
                to_remove.append(item)
            elif item.startswith('-'):
                key, value = item[1:], prompt[i+1]
                kwargs[key] = value
                to_remove.append(item)
                to_remove.append(prompt[i+1])
        for item in to_remove:
            prompt.remove(item)
        return kwargs
    
    def add_global_var(self, var_name, var):
        self.global_vars[var_name] = var

    def run_command(self, command=None, split_command=None):
        def evaluate_part(part, commands_dict):
            command = part[0]
            args_start = 1
            if command not in commands_dict:
                for dunner_command, func in DUNNER_COMMANDS.items():
                    if dunner_command in commands_dict and func(*part, **kwargs):
                        command = dunner_command
                        break
                else:
                    raise ValueError(f"{command} not in target:{commands_dict}")
                args_start = 0
            result = commands_dict[command]

            if callable(result):
                return result(*part[args_start:], global_vars=self.global_vars, **kwargs)
            elif isinstance(result, dict):
                return evaluate_part(part[1:], result)
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
                if item.isnumeric():
                    command[i] = numeric(item)
        else:
            command = split_command

        kwargs = CommandSet.extract_kwargs(command)
        return evaluate_part(command, self.command_dict)

    def __repr__(self) -> str:
        return f"{CommandSet.__name__}(pre_prompt={self.pre_prompt})"

