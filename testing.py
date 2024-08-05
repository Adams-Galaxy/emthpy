import emthpy as ep
from emthpy import fraction as frac

A = ep.matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 'a'],
    ])

B = ep.matrix([
    [0, 2, 4],
    [1, 3, 7],
    [2, 1, -5],
    ])

print(f"A:\n{type(A).__name__}")
print(f"A(1):\n{A(frac(2, 3))}")

