import random
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
import copy

import lib.expressions as Expressions # Assuming this is your custom expressions library

def init(dimensions):
    """Initializes symbols, binary and unary operations."""
    x = Expressions.Variable("x")
    v_x = Expressions.Variable("v_x")
    a_x = Expressions.Variable("a_x")
    symbols = [x, v_x, a_x]
    if dimensions > 1:
        y = Expressions.Variable("y")
        v_y = Expressions.Variable("v_y")
        a_y = Expressions.Variable("a_y")
        symbols.extend([y, v_y, a_y])  # Use extend for cleaner concatenation

    symbols_used = copy.deepcopy(symbols)
    operations_binary = [Expressions.Add, Expressions.Multiply,
                         Expressions.Subtract, Expressions.Divide]
    operations_unary = [Expressions.Sin, Expressions.Cos, Expressions.Exp] # Added sin, cos, exp
    added_constants = random.randint(0, len(symbols) + 1)
    for i in range(added_constants):
        symbols.append(Expressions.Constant(random.uniform(-10, 10)))
    return symbols, operations_binary, operations_unary, symbols_used


def get_a_single_expression(dimensions):
    symbols, operations_binary, operations_unary, symbols_used = init(
        dimensions)
    expr = create_expression(symbols, operations_binary,
                             operations_unary, symbols_used) # Assumed to exist and work
    return expr


def generate_expressions(num_of_expressions, dimensions):
    res = []
    symbols, operations_binary, operations_unary, symbols_used = init(1)
    for i in range(num_of_expressions):
        expr = get_a_single_expression(dimensions)
        res.append(expr)
    return res, symbols_used


if __name__ == '__main__':
    expr = get_a_single_expression(1)
    print(expr)