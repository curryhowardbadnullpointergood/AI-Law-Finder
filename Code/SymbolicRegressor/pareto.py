
import numpy as np
from sympy import Symbol, Integer, Float, Add, Mul, Pow, Function
import sympy

class Point:
    def __init__(self, complexity, accuracy, expression):
        self.complexity = complexity
        self.accuracy = accuracy
        self.expression = expression

    def __repr__(self):
        return f"Point(complexity={self.complexity}, accuracy={self.accuracy}, expression={self.expression})"

def calculate_complexity(expr):
    if expr.is_Symbol:
        return 1  # Variable
    elif expr.is_Number:
        return 1 if expr == 1 or expr == 0 else 2  # Constants
    elif expr.is_Add:
        return sum(calculate_complexity(arg) for arg in expr.args) + 1  # +1 for the 'Add' node
    elif expr.is_Mul:
        return sum(calculate_complexity(arg) for arg in expr.args) + 1  # +1 for the 'Mul' node
    elif expr.is_Pow:
        base, exp = expr.args
        return calculate_complexity(base) + calculate_complexity(exp) + 1  # +1 for 'Pow'
    elif isinstance(expr, Function):
        return sum(calculate_complexity(arg) for arg in expr.args) + 2  # +2 for functions like sin, log, etc.
    else:
        # Fallback: assume some complexity for unknown structures
        return 5

def dominates(a, b):
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

def update_pareto_points(current_points, new_formulas, losses):
    new_points = []
    for expr, loss in zip(new_formulas, losses):
        complexity = calculate_complexity(expr)
        new_points.append((complexity, loss, expr))

    combined = current_points + new_points

    pareto = []
    for pt in combined:
        if not any(dominates(other, pt) for other in combined if other != pt):
            pareto.append(pt)

    seen = set()
    unique_pareto = []
    for pt in pareto:
        key = (pt[0], pt[1], str(pt[2]))
        if key not in seen:
            unique_pareto.append(pt)
            seen.add(key)

    return unique_pareto


