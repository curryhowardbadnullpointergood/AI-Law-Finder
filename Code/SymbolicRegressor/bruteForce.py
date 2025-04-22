import sympy as sp 
import numpy as np 
from itertools import permutations, product, combinations


# this is the brute force file tries to brute force to fit the data 


# symbols gloabl var so that the same computation doesn't have to be run over and over 
# symbols = [] 



# this takes in a numpy array of variables and target 
def load_data(x: np.ndarray, y: np.ndarray, var_names):
    assert x.shape[1] == len(var_names)
    return x, y, [sp.Symbol(v) for v in var_names]


# generates the expression 
def generate_expressions(variables, constants, operators, max_depth):
    
    
    symbols = [sp.Symbol(v) for v in variables] + [sp.Integer(c) for c in constants]
    expressions = []
    
    for a, b in permutations(symbols, 2):
        for op in operators:
            try:
                if op == '+':
                    expressions.append(a + b)
                elif op == '-':
                    expressions.append(a - b)
                elif op == '*':
                    expressions.append(a * b)
                elif op == '/':
                    expressions.append(a / b)
                elif op == '**':
                    expressions.append(a ** b)
            except:
                continue
    return expressions




def recursive_expressions(operators, variables, constants, max_depth=2):
    symbols = [sp.Symbol(v) for v in variables]

    def build_expressions(current_depth):
        if current_depth == 0:
            return symbols.copy()

        new_expressions = []
        prev_expressions = build_expressions(current_depth - 1)

        for a in prev_expressions:
            for b in prev_expressions:
                for op in operators:
                    try:
                        if op == '+':
                            new_expressions.append(a + b)
                        elif op == '-':
                            new_expressions.append(a - b)
                        elif op == '*':
                            new_expressions.append(a * b)
                        elif op == '/':
                            new_expressions.append(a / b)
                        elif op == '**':
                            new_expressions.append(a ** b)
                    except:
                        continue

        return new_expressions

    all_exprs = build_expressions(max_depth)
    return all_exprs


# this evaluates a single expression that you pass to it 
def evaluate_expression(expression, variables, X): 
    
    symbols = [sp.Symbol(v) for v in variables]
    func = sp.lambdify(symbols, expression, modules='numpy')
    try:
        inputs = [X[:, i] for i in range(X.shape[1])]
        return func(*inputs)
    except Exception as e:
        return np.full(X.shape[0], np.nan)
    
    

# this evaluated the loss. 
def evaluate_loss(result, original):

    # mean error description length first 
    print("Calculating Mean Error Description Length: ")
    mean_error_descripton_length(result, original)


    


def mean_error_descripton_length(result, original):
    result = np.array(result).flatten()
    original = np.array(original).flatten()

    total_log_error = 0.0
    n = len(result)

    for i in range(n):
        error = abs(original[i] - result[i])
        log_error = np.log2(1 + error ** 2)
        total_log_error += log_error
        print(f"Index {i}: true={original[i]}, pred={result[i]}, error={error}, log_error={log_error:.4f}")

    return (total_log_error / n)





    
    



# exploits the symmertrical properties of physical equations in oder to even half the search space 
def symmetrical_property(expression):
    # removes duplicate expressions 
    expList = list(set(expression))
    return expList
    
# so further pruning the search space through remove all the expression if it does not contain all the variables given to it. 
# must always call the getvars method before this else it won't work 
def variable_check(expressions, variables):
    temp_vars = [sp.Symbol(v) for v in variables] 
    temp_vars = set(temp_vars)
    return [expr for expr in expressions if temp_vars.issubset(expr.free_symbols)]
    

def apply_constants(expressions, constants, max_depth=2):
    def recursive_const(expr, depth):
        results = set()
        if depth == 0:
            return results

        # Apply each constant directly to the full expr
        for const in constants:
            direct = const(expr)
            results.add(direct)
            # Nest further
            nested = recursive_const(direct, depth - 1)
            results.update(nested)

        # If it's compound, try applying constants to subparts
        if not expr.is_Atom:
            args = expr.args
            combos = product([False, True], repeat=len(args))
            for const in constants:
                for mask in combos:
                    new_args = [
                        const(arg) if use else arg
                        for use, arg in zip(mask, args)
                    ]
                    try:
                        new_expr = expr.func(*new_args)
                        results.add(new_expr)
                        # Recurse deeper on this new expression
                        results.update(recursive_const(new_expr, depth - 1))
                    except:
                        continue

        return results

    final_results = set()

    for expr in expressions:
        # Always include base expr
        final_results.add(expr)

        # Apply each constant multiplicatively
        for const in constants:
            const_expr = const(expr)
            final_results.add(const_expr)
            final_results.add(const_expr * expr)

        # Apply recursively to generate all permutations and nested
        final_results.update(recursive_const(expr, max_depth))

        # Special: handle multiplicative subparts like a * m -> sin(a)*sin(m)
        if expr.is_Mul:
            factors = expr.args
            for const in constants:
                for r in range(1, len(factors) + 1):
                    for combo in combinations(factors, r):
                        applied = [const(x) for x in combo]
                        untouched = [x for x in factors if x not in combo]
                        combined = sp.Mul(*(applied + untouched))
                        final_results.add(combined)

    return list(final_results)


def apply_constants_version2(expressions, constants, depth=1):
    def recurse(expr, current_depth):
        results = set()

        # Base: apply each constant to the full expression
        for const in constants:
            results.add(const(expr))

        if expr.is_Atom or current_depth == 0:
            return results

        args = expr.args
        combos = product([False, True], repeat=len(args))

        for mask in combos:
            new_args = []
            for use_const, arg in zip(mask, args):
                if use_const:
                    for const in constants:
                        new_args.append(const(arg))
                else:
                    # Recurse into subexpressions
                    sub_exprs = recurse(arg, current_depth - 1)
                    new_args.append(arg)
                    new_args.extend(sub_exprs)

            try:
                new_expr = expr.func(*new_args[:len(args)])
                results.add(new_expr)
            except:
                continue

        return results

    final_results = set()
    for expr in expressions:
        final_results.update(recurse(expr, depth))

    return list(final_results)


# so this is for constants 
def apply_constants_version_1(expressions, constants):
    # constants = [sp.Symbol(c) for c in constants] 
    # print(constants)
    results = set()

    for expr in expressions:
        for const in constants:
            # Always include const(expr) as multiplication-like
            results.add(const(expr))

            if expr.is_Atom:
                continue  # can't split it further

            args = expr.args
            combos = product([False, True], repeat=len(args))

            for mask in combos:
                new_args = [
                    const(arg) if use else arg
                    for use, arg in zip(mask, args)
                ]

                try:
                    new_expr = expr.func(*new_args)
                    results.add(new_expr)
                except:
                    continue

    return list(results)


# filters expressions such that only those will all inputted constants are left from a list of expressions 

def filterConstant(expressions, constants):
    def has_all_constants(expr):
        expr_str = str(expr)
        return all(const.__name__ in expr_str for const in constants)

    return [expr for expr in expressions if has_all_constants(expr)]

# gets the variables and turns it into sympy symbols 
def get_vars(variables):
    symbols = [sp.Symbol(v) for v in variables]
    return


if __name__ == "__main__":
    ops = ['*']
    vars = ['a', 'm']
    consts = []
    result = generate_expressions(vars, consts , ops, 1)
    list1 = symmetrical_property(result)
    print("###################################")
    print("Expression List generated: ")
    print(list1)
    
    # eval expression 
    print("###################################")
    print("Expression evaluated: ") 
    X = np.array([[1, 2], [3, 4], [5, 6]])
    print(evaluate_expression(list1, vars, X))

    print("###################################")
    print("Expression loss Calculating: ")
    result = np.array([[1], [3], [5]])
    original = np.array([[2], [4], [5]])
    print(mean_error_descripton_length(result, original))

    
    
    print("###################################")
    print("testing if recursive expression generation works:")
    result = recursive_expressions(ops, vars, consts, 1)
    print(result)
    print(type(result))
    print("Symmetrical property: ") 
    result_list = set(result)
    result_list2 = list(result_list)
    print(result_list2)
    print(type(result_list2))


    print("######################################")
    print("Testing the get variable symbols function:")
    # for some reason this is not working so ignore for now 
    # get_vars(vars)
    #print(symbols)
    result_list3 = variable_check(result_list2, vars)
    print(result_list3)

    print("######################################")
    print("Testing to see if constants work: ")
    #consts = ['sin']
    # well this does not work 
    consts = [sp.sin]
    print(apply_constants(result_list3, consts,1))
    consts = [sp.sin, sp.cos]
    result = apply_constants(result_list3, consts, 2)
    print(result)

    print("########################################")
    print("Testing filter constants: ")
    print(filterConstant(result, consts))



