import emthpy as ep
from _emthpy_functions import Function
func_to_str = Function.func_to_str

A = ep.matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 11],
])

print(A.inversed())
