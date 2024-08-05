from _emthpy_exceptions import InvalidCommandError
from _emthpy_functions import Function, try_numeric


class Command:
    def __init__(self, command):
        self.command = command
        self.keys = command.split()
        self.params = Command._extract_params(self.keys)
        self.current_index = 0

    @property
    def active_key(self):
        """Returns the current key"""
        return self.keys[self.current_index]

    def pop_key(self):
        """Returns the current key and increments the current index"""
        if self.current_index >= len(self.keys):
            return None
        self.current_index += 1
        return self.keys[self.current_index - 1]

    def relative_key(self, index):
        """Returns the key at the specified index relative to the current index"""
        return self.keys[self.current_index + index]
    
    def get_parameter(self, key, default):
        """Returns the value of the specified parameter"""
        return self.params.get(key, default)

    def set_parameter(self, key, value):
        """Sets the value of the specified parameter"""
        self.params[key] = value

    @staticmethod
    def _extract_params(command):
        """Extracts the parameters from the command key-list and returns them as a dictionary"""
        result = {}
        to_remove = []
        for i, key in enumerate(command):
            if key.startswith('--'):
                result[key[2:]] = True
                to_remove.append(i)
            elif key.startswith('-'):
                result[key[1:]] = try_numeric(command[i + 1])
                to_remove.append(i)
                to_remove.append(i + 1)
        to_remove.sort(reverse=True)
        for i in to_remove:
            command.pop(i)
        return result

    def __str__(self):
        return self.command

    def __next__(self):
        self.current_index += 1
        if self.current_index >= len(self.keys):
            raise StopIteration
        return self.active_key

    def __iter__(self):
        self.current_index -= 1
        return self
    
    def __getitem__(self, key):
        return self.keys[key]

class Console:
    def __init__(self, command_tree, name="Console", variable_keys=[]):
        self._command_tree = command_tree
        self._running = False
        self.name = name
        self.global_vars = {}
        self.config_vars = {}

        self.variable_keys = {
            '__assingment__': lambda command, console: len(command.keys) >= command.current_index + 3 and 
            command.relative_key(1) == '=',
            '__global_var__': lambda command, console: command.active_key in console.global_vars,
            '__config_var__': lambda command, console: command.active_key in console.config_vars,
            '__any__': lambda command, console: True,
        }
        self.variable_keys.update(variable_keys)

    def run(self, command=None, **kwargs):
        debug = kwargs.get('debug', False)

        if command is not None:
            return self.preform(command)
        
        self._running = True
        while self._running:
            str_command = input(f"{self.name}: ")
            command = Command(str_command)
            try:
                self.preform(command)
            except InvalidCommandError as e:
                print(e)
            except Exception as e:
                if debug:
                    raise e
                print(f"[Error]: {e}")

            
    def preform(self, command):
        """
        Executes the specified command.

        Args:
            command (dict): A dictionary representing the command to be executed. The keys of the dictionary
                            correspond to the steps in the command execution process.

        Returns:
            The result of executing the command, if the command is callable.

        Raises:
            InvalidCommandError: If the command is not found in the command tree or if the final step in the
                                 command execution process does not have a callable function.
        """
        if isinstance(command, str):
            command = Command(command)
        elif not isinstance(command, Command):
            raise TypeError("Command must be a string or a Command object")
        
        # If the command is empty, return
        if len(command.keys) == 0:
            return
        
        current_command = self._command_tree
        for key in command:
            while isinstance(current_command, dict) and key not in current_command:
                for var_key, is_valid in self.variable_keys.items():
                    if var_key in current_command:
                        if not is_valid(command, self):
                            continue
                        current_command = current_command[var_key]
                        break
                else:
                    print(f"Command '{key}' not found")
                    print(f"Available commands:")
                    Console._print_dict(current_command, key)
                    return
                
            if isinstance(current_command, dict):
                current_command = current_command[key]
            if isinstance(current_command, tuple):
                current_command = current_command[0] # Get the callable function from the tuple
            if callable(current_command):
                return current_command(command, self)
            
        # Reached the end of the command tree, with no callable function
        if isinstance(current_command, dict):
            Console._print_dict(current_command, command.keys[-1])
            return
        raise InvalidCommandError(f"Command '{key}' in '{command}' has no callable function")
    
    @staticmethod
    def _print_dict(dictionary, name):
        print(name + ":")
        for key, value in dictionary.items():
            if isinstance(value, tuple):
                print(f"\t{key} : {value[1]}") # Print the command usage
            else:
                print(f"\t{key}...")

    def stop(self):
        self._running = False