import time
import tracemalloc
import random
import numpy as np
from pysr import PySRRegressor
import ast
import pandas as pd
import re
import subprocess
import ast
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import symbolicRegressor as sr
import generateData as gd

def test_performance(func, *args, **kwargs):
    # Start measuring memory
    tracemalloc.start()
    start_time = time.time()

    result = func(*args, **kwargs)

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    time_taken = end_time - start_time
    memory_used = peak / 1024  # in KB

    return result, time_taken, memory_used



# number of iterations of the random noise model
modeliterationsrandom = 1; 




def varyingnoiserandomConservation():
    print("This is varying noise randomly: ")
    list_x = list(range(1,modeliterationsrandom))  
    y_results = list(map(noisy_conservation, list_x))  # This will store models
    print(y_results)  # Now prints actual model objects

    # save y to a file then later parse through it 
    with open("y_results_noise_conservation.txt", "w") as file:
        file.write(",".join(map(str, y_results)))


# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# THIS IS FOR CONSERVATION EQUATION: 


n_samples = 100
m1 = np.random.rand(n_samples,2) * 5 + 1  # Mass 1 (positive)
m2 = np.random.rand(n_samples,2) * 5 + 1  # Mass 2 (positive)
v1_initial = 2 * np.random.randn(n_samples,2)  # Initial velocity 1
v2_initial = 2 * np.random.randn(n_samples,2)  # Initial velocity 2



def noisy_conservation(ran): 
    
    #  Simulate a perfectly elastic collision for simplicity (momentum is conserved)
    v1_final = ((m1 - m2) / (m1 + m2)) * v1_initial + ((2 * m2) / (m1 + m2)) * v2_initial + (5* ran)
    v2_final = ((2 * m1) / (m1 + m2)) * v1_initial + ((m2 - m1) / (m1 + m2)) * v2_initial+ (5* ran)


    #Target Variable
    y = m1 * v1_initial + m2 * v2_initial + (5* ran) 

    X = np.concatenate([m1, m2, v1_initial, v2_initial, v1_final, v2_final], axis=1)


    model = PySRRegressor(
        niterations=100,  # Might need more iterations for this complex equation.
        binary_operators=["+", "*", "-"],
        unary_operators=[],
        extra_sympy_mappings={},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        
    )

    model.fit(X, y)

    return model












# Example functions to compare
def function_a(n):
    varyingnoiserandomConservation()

    return True 

def function_b(n):
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
    sr.solve_general(data, target, variables, coefficients, operators, constants, power, indepVars, tarVars, degree, epchos)


    return True






        

# Test parameters
n = 10**6

# Run and compare
result_a, time_a, mem_a = test_performance(function_a, n)
result_b, time_b, mem_b = test_performance(function_b, n)

print(f"Function A: Time = {time_a:.4f} sec, Peak Memory = {mem_a:.2f} KB")
print(f"Function B: Time = {time_b:.4f} sec, Peak Memory = {mem_b:.2f} KB")

