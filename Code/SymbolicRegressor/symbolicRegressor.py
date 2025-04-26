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
    


        symbolic_p, symbolic_U = da.symbolicTransformation(independentVars, p, U)
        print(f"Symbolic Scaling Part (Π xᵢ^pᵢ): {symbolic_p}")
        print("##############################################################################")
        
        if (len(constants)) == 1:
            print("Symbolic Result:")
            print("##############################################################################")
            #print(targetVars + '=' +  symbolic_U)
            print(targetVars + ' = ' )
            print(constants[0] * symbolic_p) 
        else: 
            print("Symbolic Result:")
            print("##############################################################################")
            #print(targetVars + '=' +  symbolic_U)
            print(targetVars + ' = ' )
            print(symbolic_p) 

        return 
        
    else:
        print(f"  Found {num_dimless_vars} dimensionless group(s), continuing...\n")
        print("##############################################################################")
    
        symbolic_p, symbolic_U = da.symbolicTransformation(independentVars, p, U)
        print(f"  Scaling Term: {symbolic_p}")
        print(f"  Dimensionless Groups: {symbolic_U}")
        print("##############################################################################")
    

    print("Preparing dimensionless data: ")
    print("##############################################################################")
    
    
    data_x_prime, data_y_prime = generate_dimensionless_data(data, target, p, U)
    print("Traning Neural Network on dimensionless data...")
    print("##############################################################################")
    
    

    train_loader, val_loader = nn.prepare_data(data_x_prime.T, data_y_prime, batch_size=2)
    model = nn.SymbolicNetwork(n_input=data_x_prime.shape[0], n_output=1)
    nn.train_network(model, train_loader, val_loader, epochs, learning_rate=1e-3, device='cpu')

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

 




def solve_double_pendulum( data, target, variables, operators, constants, power, independentVars, targetVars, degree, epchos):
    # this attempts to solve the expression: 
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
    exEval = bf.evaluate_expressions(exFilPowers, variables, result, target )
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
    'g':           [0, 1, -2, 0, 0, 0, 0],  
}



print("hello world")
# f = ma ? 
data, target = gd.generate_force_data()
variables = ['a', 'm']
coefficients = []
operators = ['*']
constants = []
power = []
indepVars = ['mass', 'acceleration']
tarVars = 'force'
degree = 2 
epchos = 50 
solve_general(data, target, variables, coefficients, operators, constants, power, indepVars, tarVars, degree, epchos)

# pendulum 
data, target = gd.generate_force_data()
variables = ['l', 'g']
coefficients = []
operators = ['*']
two_pi = 2 * sp.pi 
constants = [two_pi]
power = [0.5]
indepVars = ['length', 'g']
tarVars = 'time'
degree = 2 
epchos = 50 
solve_general(data, target, variables, coefficients, operators, constants, power, indepVars, tarVars, degree, epchos)

# double pendulum 


# defining symbols 
theta1, theta2 = sp.symbols('theta1 theta2')        
theta1_dot, theta2_dot = sp.symbols('theta1_dot theta2_dot') 
theta1_ddot, theta2_ddot = sp.symbols('theta1_ddot theta2_ddot') 
l1, l2 = sp.symbols('l1 l2')      
m1, m2 = sp.symbols('m1 m2')     
g = sp.symbols('g')

variables_eq1 = [
    'theta1', 'theta2', 'theta2_dot', 'theta1_ddot', 'theta2_ddot', 'l1', 'l2', 'm1', 'm2', 'g'
]

variables_eq2 = [
    'theta1', 'theta2', 'theta1_dot', 'theta1_ddot', 'theta2_ddot', 'l1', 'l2', 'm2', 'g'
]

n_samples = 40

theta1 = np.random.uniform(-np.pi, np.pi, n_samples)
theta2 = np.random.uniform(-np.pi, np.pi, n_samples)
theta1_dot = np.random.uniform(-5, 5, n_samples)
theta2_dot = np.random.uniform(-5, 5, n_samples)
theta1_ddot = np.random.uniform(-10, 10, n_samples)  
theta2_ddot = np.random.uniform(-10, 10, n_samples)
l1 = np.random.uniform(0.5, 2.0, n_samples)
l2 = np.random.uniform(0.5, 2.0, n_samples)
m1 = np.random.uniform(0.5, 5.0, n_samples)
m2 = np.random.uniform(0.5, 5.0, n_samples)
g = np.full(n_samples, 9.81)  # constant gravity

# === Step 2: Build x1 (input for first equation) ===
x1 = np.stack([
    theta1, theta2, theta2_dot, theta1_ddot, theta2_ddot,
    l1, l2, m1, m2, g
], axis=1)

# === Step 3: Compute y1 (target for first equation) ===
y1 = (
    (m1 + m2)*l1*theta1_ddot
    + m2*l2*theta2_ddot*np.cos(theta1 - theta2)
    + m2*l2*theta2_dot**2*np.sin(theta1 - theta2)
    + (m1 + m2)*g*np.sin(theta1)
)

# === Step 4: Build x2 (input for second equation) ===
x2 = np.stack([
    theta1, theta2, theta1_dot, theta1_ddot, theta2_ddot,
    l1, l2, m2, g
], axis=1)

# === Step 5: Compute y2 (target for second equation) ===
y2 = (
    m2*l2*theta2_ddot
    + m2*l1*theta1_ddot*np.cos(theta1 - theta2)
    - m2*l1*theta1_dot**2*np.sin(theta1 - theta2)
    + m2*g*np.sin(theta2)
)

print("DOUBLE PENDULUM: ")
operators = ['+', '-', '*']
degree = 4
constants = []
variables_simple = ['theta1', 'theta2', 'theta2_dot', 'theta1_ddot', 'theta2_ddot'] 
ex = bf.fast_recursive_expressions(operators, variables_simple, constants, degree)
print(ex)
exFiltered = bf.symmetrical_property(ex)
print("Filtered: ")
print(exFiltered)
exFilVar = bf.variable_check(exFiltered, variables_eq1)
print("Variables filter: ")
print(exFilVar)

print("-")
