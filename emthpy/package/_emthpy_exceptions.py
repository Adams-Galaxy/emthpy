"""Exceptions for emthpy"""

# ------------------------- Vectors -------------------------
class RSpaceError(Exception):
    """Exception raised for errors related to R space."""
    def __init__(self, *args: object) -> None:
        super().__init__(*args)



# ------------------------- Matrices -------------------------
class MatrixShapeError(Exception):
    """Exception raised for errors related to matrix shape."""
    def __init__(self, *args: object) -> None:
        super().__init__(*args)



# ------------------------- Functions -------------------------
class InvalidArgumentError(Exception):
    """Exception raised for invalid function arguments."""
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class UndefingedVariableError(Exception):
    """Exception raised for undefined variables."""
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class UnkownOperatorError(Exception):
    """Exception raised for unknown operators."""
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class InvalidExpressionError(Exception):
    """Exception raised for invalid equations."""
    def __init__(self, *args: object) -> None:
        super().__init__(*args)



# ------------------------- Utils -------------------------
class InvalidCommandError(Exception):
    """Exception raised for invalid commands."""
    def __init__(self, *args: object) -> None:
        super().__init__(*args)