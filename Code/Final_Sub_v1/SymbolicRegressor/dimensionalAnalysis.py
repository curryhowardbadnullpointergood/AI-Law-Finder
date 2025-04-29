import numpy as np 
import sympy as sp



UNIT_TABLE = {

    'mass':                 [1, 0, 0, 0, 0, 0, 0], 
    'length':               [0, 1, 0, 0, 0, 0, 0],
    'time':                 [0, 0, 1, 0, 0, 0, 0],
    'temperature':          [0, 0, 0, 1, 0, 0, 0],
    'current':              [0, 0, 0, 0, 1, 0, 0],
    'amount':               [0, 0, 0, 0, 0, 1, 0], 
    'luminous_intensity':   [0, 0, 0, 0, 0, 0, 1],


    'area':                   [0, 2, 0, 0, 0, 0, 0],   
    'acceleration':           [0, 1, -2, 0, 0, 0, 0],  
    'velocity':               [0, 1, -1, 0, 0, 0, 0],  
    'force':                  [1, 1, -2, 0, 0, 0, 0],  
    'energy':                 [1, 2, -2, 0, 0, 0, 0],  
    'power':                  [1, 2, -3, 0, 0, 0, 0], 
    'pressure':               [1, -1, -2, 0, 0, 0, 0],
    'charge':                 [0, 0, 1, 0, 1, 0, 0],  
    'voltage':                [1, 2, -3, 0, -1, 0, 0],
    'resistance':             [1, 2, -3, 0, -2, 0, 0],
    'capacitance':            [-1, -2, 4, 0, 2, 0, 0],
    'inductance':             [1, 2, -2, 0, -2, 0, 0], 
    'current':                [0, 0, 0, 0, 1, 0, 0],   
    'potential':              [1, 2, -3, 0, -1, 0, 0], 
    'magnetic_field':         [1, 0, -2, 0, -1, 0, 0], 
    'magnetic_flux':          [1, 2, -2, 0, -1, 0, 0], 
    'electric_field':         [1, 1, -3, 0, -1, 0, 0], 
    'permittivity':           [-1, -3, 4, 0, 2, 0, 0],
    'permeability':           [1, 1, -2, 0, -2, 0, 0],  
    'conductance':            [-1, -2, 3, 0, 2, 0, 0], 
    'density':                [1, -3, 0, 0, 0, 0, 0],   
    'frequency':              [0, 0, -1, 0, 0, 0, 0],   
    'wavenumber':             [0, -1, 0, 0, 0, 0, 0],   
    'momentum':               [1, 1, -1, 0, 0, 0, 0],   
    'angular_momentum':       [1, 2, -1, 0, 0, 0, 0],   
    'torque':                 [1, 2, -2, 0, 0, 0, 0],
    'specific_heat':          [0, 2, -2, -1, 0, 0, 0], 
    'thermal_conductivity':   [1, 1, -3, -1, 0, 0, 0], 
    'boltzmann_constant':     [1, 2, -2, -1, 0, 0, 0], 
    'entropy':                [1, 2, -2, -1, 0, 0, 0],  
    'temperature':            [0, 0, 0, 1, 0, 0, 0],    
    'time':                   [0, 0, 1, 0, 0, 0, 0],    
    'length':                 [0, 1, 0, 0, 0, 0, 0],    
    'mass':                   [1, 0, 0, 0, 0, 0, 0],    
    'volume':                 [0, 3, 0, 0, 0, 0, 0],    
    'surface_charge_density': [0, -2, 1, 0, 1, 0, 0],   
    'volume_charge_density':  [0, -3, 1, 0, 1, 0, 0],   
    'current_density':        [0, -2, 0, 0, 1, 0, 0],   
    'light_intensity':        [0, 0, 0, 0, 0, 0, 1],    
    'amount':                 [0, 0, 0, 0, 0, 1, 0],    
    'luminous_intensity':     [0, 0, 0, 0, 0, 0, 1],    
    'dimensionless':          [0, 0, 0, 0, 0, 0, 0],    
}




def get_matrix_target(independent_vars, dependent_var, units_dict):
    try:
        M = np.column_stack([units_dict[var.lower()] for var in independent_vars])
    except KeyError as e:
        raise ValueError(f"Independent variable '{e.args[0]}' not found in unit dictionary.")

    try:
        b = np.array(units_dict[dependent_var.lower()])
    except KeyError:
        raise ValueError(f"Dependent variable '{dependent_var}' not found in unit dictionary.")

    return M, b  




def solveDimension(M, b):
    M_sym = sp.Matrix(M)
    b_sym = sp.Matrix(b)

    try:
        p = M_sym.LUsolve(b_sym)
    except Exception as e:
        raise ValueError(f"Failed to solve Mp = b: {e}")

    U = M_sym.nullspace()

    return p, U



def generate_dimensionless_data(data_x, data_y, p, U):

    if not isinstance(p, np.ndarray):
        p = np.array(p).astype(np.float64).flatten()

    p = p.reshape(-1, 1)
    scaling_factor = np.prod(np.float_power(data_x, p), axis=0)
    
    data_y_prime = data_y / scaling_factor

    dimensionless_vars = []
    
    if U:
        dimensionless_vars = []
        for u in U:
            new_var = np.prod(np.float_power(data_x, u), axis=0)
            dimensionless_vars.append(new_var)

        data_x_prime = np.vstack(dimensionless_vars)
    else:
        data_x_prime = data_x  # No transformation if no dimensionless groups

    return data_x_prime, data_y_prime


def symbolicTransformation(independent_vars, p, U):
    symbols = [sp.symbols(var) for var in independent_vars]
    
    symbolic_p = sp.Mul(*[symbols[i]**p[i] for i in range(len(independent_vars))])
    
    symbolic_U = []
    for u in U: 
        expr_u = sp.Mul(*[symbols[i]**u[i] for i in range(len(independent_vars))])
        symbolic_U.append(expr_u)
    
    return symbolic_p, symbolic_U
