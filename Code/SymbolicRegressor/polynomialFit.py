import numpy as np 
import sympy as sp
import itertools
# sanity check to make sure data array and variables are legit
def load_data(data_array, variables):

    num_cols = data_array.shape[1]
    num_vars = len(variables)

    if num_cols != num_vars:
        raise ValueError(f"Data has {num_data_cols} columns but {num_vars} variables were provided.")

    return data_array


# generate polynimial expressions
def generate_expressions(coeffs, variables, operators, max_degree):

    vars_syms = [sp.Symbol(v) for v in variables]

    expressions = []


    for powers in itertools.product(range(1, max_degree + 1), repeat=len(vars_syms)):
        terms = [c * (v ** p) for c, v, p in zip(coeffs, vars_syms, powers)]
        for ops in itertools.product(operators, repeat=len(terms) - 1):
            expr = terms[0]
            for op, term in zip(ops, terms[1:]):
                if op == '+':
                    expr = expr + term
                elif op == '-':
                    expr = expr - term
                elif op == '*':
                    expr = expr * term
                elif op == '/':
                    expr = expr / term
            expressions.append(sp.simplify(expr))

    return expressions






def filter_expressions(expressions, required_vars, required_constants, required_power):
    result = []

    # Convert variable names to sympy Symbols
    required_vars = {sp.Symbol(v) for v in required_vars}

    for expr in expressions:
        symbols_ok = required_vars.issubset(expr.free_symbols)

        if not all(
            any(
                (isinstance(sub, const) if isinstance(const, type)
                 else sub == const)
                for sub in sp.preorder_traversal(expr)
            )
            for const in required_constants
        ):
            continue


        power_ok = any(
            isinstance(sub, sp.Pow) and sub.exp == required_power
            for sub in sp.preorder_traversal(expr)
        )

        if symbols_ok and power_ok:
            result.append(expr)

    return result


#def filter_expressions(expr_list, required_vars, required_constants, required_power):
#    filtered = []

#    required_vars #= set(sp.Symbol(v) for v in required_vars)
#    for expr in expr_list:
#        vars_in_expr = expr.free_symbols
#        if not required_vars.issubset(vars_in_expr):
#            continue

  #      if any(not any(isinstance(sub, const) for sub in sp.preorder_traversal(expr)) for const in required_constants):
   #         continue

        
    #    has_required_power = any(
  #          isinstance(sub, sp.Pow) and sub.exp == required_power
     #       for sub in expr.preorder_traversal()
      #  )
       # if not has_required_power:
        #    continue
#
 #       filtered.append(expr)
#
 #   return filtered


def evaluate_expressions(expressions, variables, data, y_true):
    results = []
    for expr in expressions:
        try:
            func = sp.lambdify(variables, expr, modules='numpy')
            y_pred = np.array([func(*row) for row in data])
            rmse = np.sqrt(np.mean((y_pred - y_true)**2))
            results.append((expr, rmse))
        except Exception as e:
            print(f"Skipping {expr} due to error: {e}")
    return results


def evaluate_expressions_old2(data, y_true, expr, variables):
    
    func = sp.lambdify(variables, expr, modules='numpy')

    y_pred = np.array([func(*row) for row in data])

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return rmse 


def evaluate_expressions_old(expressions, variables, X_data, y_data):
    losses = []
    best_expr = None
    best_loss = float("inf")

    for expr in expressions:
        f_lambdified = sp.lambdify(variables, expr, modules=["numpy"])
        try:
            y_pred = f_lambdified(*[X_data[:, i] for i in range(X_data.shape[1])])
            y_pred = np.array(y_pred, dtype=float).squeeze()
        except Exception as e:
            # If the expression fails to evaluate, assign infinite loss
            losses.append(float("inf"))
            continue

        loss = np.sqrt(np.mean((y_pred - y_data) ** 2)) 
        losses.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_expr = expr

    return losses, best_expr



def bestFit(results):
    return min(results, key=lambda x: x[1])

