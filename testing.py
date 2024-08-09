import emthpy as ep
import numpy as np

A = ep.matrix([
    [0, 3, 1],
    [1, 1, '2k'],
    [3, 2, 4],
])

B = ep.matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 11],
])

C = ep.matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 'k'],
])

#print(B.inversed())
print(C(11))
print(C(11).inversed())
#print(C(11).solve(ep.vector([1, 2, 3])))
