#import symbolicRegressor as sR 
import numpy as np 
import time
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





# U = m . g . z no 16 

np.random.seed(0)  


m = np.random.uniform(1, 10, 100)   
g = np.random.uniform(9.7, 9.9, 100)  
z = np.random.uniform(0, 100, 100)  


data = np.column_stack((m, g, z))


target = (m * g * z).reshape(-1, 1)

variables = ['m', 'g', 'z']
coefficients = [1,1,1]
operators = ['*']
constants = []
power = []
indepVars = ['mass', 'g', 'length']
tarVars = 'g'
degree = 2
epochs = 50
start_time = time.time()
r = pf.generate_expressions(coefficients, variables, operators, 2)
re = pf.evaluate_expressions(r, variables, data, target)
print(re)
bestf = pf.bestFit(re)
print("Best fit")
print(bestf)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# 1/2*k_spring*x**2 = u no 17

np.random.seed(1)  


k = np.random.uniform(10, 1000, 100)  
x = np.random.uniform(-2, 2, 100)


data = np.column_stack((k, x))


target = (0.5 * k * x**2).reshape(-1, 1)

variables = ['k_spring', 'x']
power = [2]
coefficients = [1,1]
operators = ['*']
constants = [0.5,1]
start_time = time.time()
r = bf.generate_expressions(variables, constants, operators, 1)
#rr = bf.apply_powers(r, power, 1)
rrr = bf.apply_constants_fully_symmetric(r, constants, 1)
print(rrr)
end_time = time.time()
execution_time = end_time - start_time

# 26 q/C = volt 

np.random.seed(2)  


q = np.random.uniform(1, 100, 100)   
C = np.random.uniform(1, 100, 100)   


X = np.column_stack((q, C))


y = (q / C).reshape(-1, 1)


# omega/c = k no 29 

import numpy as np

np.random.seed(3)

omega = np.random.uniform(1, 100, 100)
c = np.random.uniform(1, 100, 100)

X = np.column_stack((omega, c))

y = (omega / c).reshape(-1, 1)


# omega = q*v*B/p no 35 

np.random.seed(4)

q = np.random.uniform(1, 100, 100)
v = np.random.uniform(1, 100, 100)
B = np.random.uniform(1, 10, 100)
p = np.random.uniform(1, 10, 100)

X = np.column_stack((q, v, B, p))

y = (q * v * B / p).reshape(-1, 1)

# omega_0/(1-v/c) = omega   no 36 

np.random.seed(5)

omega_0 = np.random.uniform(1, 100, 100)
v = np.random.uniform(1, 99, 100)
c = np.random.uniform(100, 300, 100)

X = np.column_stack((omega_0, v, c))

y = (omega_0 / (1 - v / c)).reshape(-1, 1)


# E_n =(h/(2*pi))*omega     no 38 


np.random.seed(6)

h = np.random.uniform(1, 10, 100)
omega = np.random.uniform(1, 100, 100)

X = np.column_stack((h, omega))

y = (h / (2 * np.pi) * omega).reshape(-1, 1)


# e_n = 3/2*pr*V  no 41 
np.random.seed(7)

p_r = np.random.uniform(1, 100, 100)
V = np.random.uniform(1, 100, 100)

X = np.column_stack((p_r, V))

y = (3 / 2 * p_r * V).reshape(-1, 1)


# d = mob*kb*T no 47 

np.random.seed(8)

mob = np.random.uniform(1, 10, 100)
k_b = np.random.uniform(1, 2, 100)
T = np.random.uniform(100, 300, 100)

X = np.column_stack((mob, k_b, T))

y = (mob * k_b * T).reshape(-1, 1)


# e_den = epsilon*Ef**2 no 75 

np.random.seed(9)

epsilon = np.random.uniform(1, 10, 100)
Ef = np.random.uniform(1, 100, 100)

X = np.column_stack((epsilon, Ef))

y = (epsilon * Ef**2).reshape(-1, 1)




