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



x = np.array([
    [5, 5, 3, 2],
    [6, 4, 5, 5],
    [10, 8, 5, 7],
    [3, 3, 2, 1],
    [12, 10, 8, 6],
    [2, 2, 1, 1],
    [7, 6, 5, 4],
    [4, 4, 3, 3],
])

y = 2 * (x[:, 0] + x[:, 1]) + 4 * (x[:, 2] + x[:, 3])

variables = ['A', 'T', 'G', 'C']

coefficients = [2, 2 , 4, 4]

operators = ['+']

max_degree = 2 

r = pf.generate_expressions(coefficients, variables, operators, 1)
print(r)
#re = pf.filter_expressions(r, variables, coefficients, 1)
ree = pf.evaluate_expressions(r, variables, x, y )
print("eval")
print(ree)
