import emthpy as ep
import math

from os import system
system("clear")

f = ep.function("2x^2+3")
g = ep.function("5x/3")

print(f"f: {f}")
print(f"g: {g}")

print(f"\nf(2): {f(2)}")
print(f"g(9): {g(9)}")
print(f"f(g(9)): {f(g(9))}")
print(f"g(f(2)): {g(f(2))}")
print(f"\ng(f): {g(f)}")

h = ep.function("sin(t_hyp)")

print(f"\nh: {h}")
print(f"h(pi/2): {h(math.pi/2)}")

l = ep.function("ln(e^x)")
print(f"\nl: {l}")
print(f"l(5): {l(5)}")

A = ep.matrix([
    [1, 2],
    [3, 4],
])

B = ep.matrix([
    [1, ep.function("3a")],
    [3, ep.function("a/2")],
])

print(f"\nA:\n{A}")
print(f"\nB:\n{B}")
print(f"\nA+B:\n{A+B}")
print(f"\nA+B|a=2:\n{(A + B).evaluated(a=1)}")

C = ep.matrix([
    [1, 2, 3],
    [0, 1, 4],
    [5, 6, 0],
])

print(f"\nC:\n{C}")
print(f"\nC^T:\n{C.T}")
print(f"\nC inversed:\n{C.inversed()}")
print(f"\nC*C:\n{C@C}")

p = ep.vector([1, 2, 3])
q = ep.vector([4, 5, 6])

print(f"\np: {p}")
print(f"q: {q}")
print(f"\np+q: {p+q}")
print(f"\np*q: {p*q}")
print(f"\np.q: {p@q}")
