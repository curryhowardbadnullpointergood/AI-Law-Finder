import numpy as np
from sympy import Matrix, symbols, S, N
from scipy.linalg import lstsq

def getPowers(input_units, output_units):
    """
    Solves for exponents that make the product of input units equal to the output unit.
    """
    input_units = np.array(input_units)           # shape: (num_units, num_vars)
    output_units = np.array(output_units).flatten()  # shape: (num_units,)

    if input_units.shape[0] != output_units.shape[0]:
        raise ValueError(f"Dimension mismatch: input_units is {input_units.shape}, output_units is {output_units.shape}")

    try:
        powers = np.linalg.lstsq(input_units, output_units, rcond=None)[0]
        return powers
    except Exception as e:
        raise ValueError(f"Failed to solve for powers: {e}")

def compute_nullspace(matrix):
    return Matrix(matrix).nullspace()

def apply_dimensional_analysis(X, y, var_names, units_db):
    """
    Apply dimensional analysis to reduce variables to dimensionless forms.

    Parameters:
    - X (np.ndarray): Input variables (shape: n_samples x n_variables)
    - y (np.ndarray): Target variable (shape: n_samples,)
    - var_names (list of str): Variable names corresponding to X columns
    - units_db (dict): Mapping from variable names to unit vectors

    Returns:
    - transformed_X: np.ndarray of dimensionless input variables
    - transformed_y: np.ndarray of dimensionless target variable
    - new_var_names: list of strings for new Pi terms
    - solved_part_expr: SymPy expression of solved units
    - new_var_exprs: List of SymPy expressions for new variables
    """
    input_units = np.array([units_db[v] for v in var_names]).T
    target_units = np.array(units_db['y']).reshape(-1, 1)  # assuming output is called 'y'

    # Check if all variables are already dimensionless
    if not input_units.any():
        input_syms = symbols(var_names)
        return X, y, var_names, S(1), list(input_syms)

    # Step 1: Solve for dimensional consistency
    M = Matrix(input_units)
    solved_powers = getPowers(input_units, target_units)
    solved_syms = symbols(var_names)
    solved_expr = S(1)
    for i, p in enumerate(solved_powers):
        solved_expr *= solved_syms[i] ** round(p, 2)

    # Step 2: Compute nullspace for dimensionless groups (Pi terms)
    nullspace = compute_nullspace(input_units)
    new_var_exprs = []
    for vec in nullspace:
        expr = S(1)
        for i in range(len(vec)):
            expr *= solved_syms[i] ** vec[i]
        new_var_exprs.append(expr)

    # Step 3: Apply transformations to X and y
    func = np.ones_like(y, dtype=np.float64)
    for i in range(len(solved_powers)):
        func *= X[:, i] ** solved_powers[i]
    transformed_y = y / func

    new_vars = []
    for vec in nullspace:
        dimless = np.ones(X.shape[0], dtype=np.float64)
        for j in range(len(vec)):
            # Ensure vec[j] is a numeric value
            dimless *= X[:, j] ** float(vec[j]) if isinstance(vec[j], (int, float)) else N(X[:, j] ** vec[j])
        new_vars.append(dimless)

    if new_vars:
        transformed_X = np.vstack(new_vars).T
        new_var_names = [f"pi{i+1}" for i in range(len(new_vars))]
    else:
        transformed_X = np.empty((X.shape[0], 0))  # no new vars
        new_var_names = []

    return transformed_X, transformed_y, new_var_names, solved_expr, new_var_exprs

# Example to test:
X = np.array([[1, 2], [2, 4], [3, 6]], dtype=np.float64)
y = np.array([3, 6, 9], dtype=np.float64)
var_names = ['x1', 'x2']
units_db = {
    'x1': [1, 0, 0, 0, 0, 0],  # Example unit: m
    'x2': [0, 1, 0, 0, 0, 0],  # Example unit: s
    'y': [0, 0, 1, 0, 0, 0]    # Example unit: kg
}

# Applying dimensional analysis
transformed_X, transformed_y, new_var_names, solved_expr, new_var_exprs = apply_dimensional_analysis(X, y, var_names, units_db)
print("Transformed X:", transformed_X)
print("Transformed y:", transformed_y)
print("New variable names:", new_var_names)
print("Solved part expression:", solved_expr)
print("New variable expressions:", new_var_exprs)



