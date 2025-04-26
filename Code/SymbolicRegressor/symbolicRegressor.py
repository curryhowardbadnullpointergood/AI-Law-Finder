import numpy as np 
import polynomialFit as pf 
import sympy as sp 
import numpy as np
from sympy import symbols, simplify, S
import dimensionalAnalysis as da 
import neuralNetwork as nn 
from pareto import Point 
import pareto as p 
import plots as plots
import generateData as gd 
import bruteForce as bf 



# main symbolic regression loop, for f = ma 

def solveFma():
    # generating data 
    x, y = gd.generate_force_data()
    # dimensional analysis 
    independent = ['mass', 'acceleration']
    dependent = 'force'
    M, b = da.get_matrix_target(independent, dependent, UNIT_TABLE)
    # solving dimensions 
    p, U = da.solveDimension(M, b)
    

def solve_general( data, target, variables, coefficients, operators, constants, power, independentVars, targetVars, degree, epchos):
    # this attempts to solve the expression: 
    # starting with dimensional Analysis 
    print("##############################################################################")
    print("Trying Dimensional Analysis: ")
    print("##############################################################################")
    
    M, b = da.get_matrix_target(independentVars, targetVars, UNIT_TABLE)
    p,U = da.solveDimension(M,b)

    if hasattr(U, '__len__'):
        num_dimless_vars = len(U)
    else:
        num_dimless_vars = 0



    if num_dimless_vars == 0:
        print("  DA SOLVED: Result is constant * scaling.")
        print("##############################################################################")
    
        
        symbolic_p, symbolic_U = symbolicTransformation(independentVars, p, U)
        print(f"Symbolic Scaling Part (Π xᵢ^pᵢ): {symbolic_p}")
        print("##############################################################################")
    
        print("Symbolic Dimensionless Groups (Π xᵢ^uᵢⱼ):")
        print("##############################################################################")
    
        for u in symbolic_U:
            print(u)
        return 
    else:
        print(f"  Found {num_dimless_vars} dimensionless group(s), continuing...\n")
        print("##############################################################################")
    
        symbolic_p, symbolic_U = symbolicTransformation(independentVars, p, U)
        print(f"  Scaling Term: {symbolic_p}")
        print(f"  Dimensionless Groups: {symbolic_U}")
        `print("##############################################################################")
    

    print("Preparing dimensionless data: ")
    print("##############################################################################")
    
    
    data_x_prime, data_y_prime = generate_dimensionless_data(data, target, p, U)
    print("Traning Neural Network on dimensionless data...")
    print("##############################################################################")
    
    

    train_loader, val_loader = nn.prepare_data(data_x_prime.T, data_y_prime, batch_size=2)
    model = nn.SymbolicNetwork(n_input=data_x_prime.shape[0], n_output=1)
    nn.train_network(model, train_loader, val_loader, epochs=600, learning_rate=1e-3, device='cpu')

    predictions = nn.predict(model, data_x_prime.T, device='cpu')
    gradients = nn.get_gradient(model, data_x_prime.T, device='cpu')
    
    print("##############################################################################")
    print("Predictions: ")
    print(predictions)
    print("##############################################################################")
    
    print("Gradients: ")
    print(gradients)
    print("##############################################################################")

    mse = mean_squared_error(data_y_prime.flatten(), predictions.flatten())
    print(f"Final MSE on dimensionless target: {mse:.6f}")
    print("##############################################################################")

    # potential to add some plots/graphs here later 
    
    print("\nStep 3: Polynomial Fit...")
    print("##############################################################################")
    r = pf.load_data(data,variables)
    print(r)
    print("##############################################################################")
    re = pf.generate_expressions(coefficients, variables, operators,degree)
    print(f"  Generated {len(re)} polynomial candidates.")
    print(re)
    print("##############################################################################")
    ree = pf.filter_expressions(re, variables, coefficients, degree)
    print(f"  Filtered {len(ree)} polynomial expressions.")
    print(ree)
    print("##############################################################################")
    print("Evaluating polynomial expressions: ")
    print("##############################################################################")
    reee = pf.evaluate_expressions(reee, variables, data, target)
    print(reee)
    print("##############################################################################")
    print("Findings Best Fit")
    cc = pf.bestFit(reee)
    print("##############################################################################")
    print(cc)
    expression, score = cc
    if float(score) == 0.0:
        print("Solved Expression:", expression)
        return 

    print("##############################################################################")
    print("\nStep 4: Brute Force Search...")
    print("##############################################################################")
    print("Brute Force Generating Expressions: ")
    ex = bf.generate_expressions(variables, constants, operators, degree)
    print(ex)
    print("##############################################################################")
    print("Filtering using the symmetrical property: ")
    exFiltered = bf.symmetrical_property(ex)
    print(exFiltered)
    print("##############################################################################")
    print("Filtering based on variables: ")
    exFilVar = bf.variable_check(exFiltered, variables)
    print(exFilVar)


    print("##############################################################################")
    print("Applying Powers: ")
    exPow = bf.apply_powers(exFilVar, power)
    print(exPow)
    print("##############################################################################")
    print("Filtering based on Powers: ")
    exFilPow = bf.filter_powers(expow, power)
    

    print("##############################################################################")
    print("Applying constants: ")
    exFilVarCon = bf.apply_constants_fully_symmetric(exFilPow, constants, degree)
    print(exFilVarCon)
    print("##############################################################################")
    print("Filter based on constants: ")
    exFilVarConst = bf.filterConstant(exFilVarCon, constants)
    # chaining this so need to filter powers again!
    exFilPowers = bf.filter_powers(exFilVarConst, power)
    print(exFilPowers)
    print("##############################################################################")
    print("Evaluating Expressions: ")
    exEval = bf.evaluate_expressions(variables, )
    print(exEval)
    print("Finished! ")

 



# units table
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



print("hello world")
# f = ma ? 
data, target = gd.generate_force_data()
variables = ['a', 'm']
coefficients = []
operators = ['*']
constants = []
power = []
indepVars = [] 
tarVars = [] 
solve_general(data, target, variables, coefficients, operators, constants, power,   )
