"""Exceptions for emthpy"""

class RSpaceError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        
class MatrixShapeError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)