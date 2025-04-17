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

# number of iterations of the random noise model
modeliterationsrandom = 2; 




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


# this varys the noise for the conservation model
varyingnoiserandomConservation()


######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
def varyingnoiserandomNewtonSecondLaw():
    print("This is varying noise randomly: ")
    list_x = list(range(1,modeliterationsrandom))  
    y_results = list(map(noisy_fma, list_x))  # This will store models
    print(y_results)  # Now prints actual model objects

    # save y to a file then later parse through it 
    with open("y_results_noise_Newtons_Second_Law.txt", "w") as file:
        file.write(",".join(map(str, y_results)))

np.random.seed(0)  # For reproducibility
n_samples = 100

mass = np.random.uniform(1, 10, size=n_samples)  
acceleration = np.random.uniform(0, 20, size=n_samples) 

force = mass * acceleration  

X = np.column_stack((mass, acceleration))
y = force



def noisy_fma(x):

    force = mass * acceleration + ( 5*x)
    X = np.column_stack((mass, acceleration))
    y = force
    
    
    model = PySRRegressor(
        niterations=100,       # More iterations = better chance of recovery
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        populations=10,
        progress=True,
        model_selection="best",  # Simplest model that fits best
        variable_names=["m", "a"]
    )

    model.fit(X, y)
    return model 


varyingnoiserandomNewtonSecondLaw()

######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################


# Newtons First Law


np.random.seed(1)
n_samples = 1000

v = np.random.uniform(1, 10, size=n_samples)   
t = np.random.uniform(0, 10, size=n_samples)   
x0 = np.random.uniform(-5, 5, size=n_samples)  

#x = v * t + x0  


#X = np.column_stack((v, t, x0))
#y = x



def varyingnoiserandomNewtonFirstLaw():
    print("This is varying noise randomly: ")
    list_x = list(range(1,modeliterationsrandom))  
    y_results = list(map(noisy_fma, list_x))  # This will store models
    print(y_results)  # Now prints actual model objects

    # save y to a file then later parse through it 
    with open("y_results_noise_Newtons_First_Law.txt", "w") as file:
        file.write(",".join(map(str, y_results)))




def noisy_fma(ran):

    x = v * t + x0 + (5*ran)
    X = np.column_stack((v,t,x0))
    y = x 


    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        populations=10,
        progress=True,
        model_selection="best",
        variable_names=["v", "t", "x0"]
    )

    model.fit(X, y)
    return model



varyingnoiserandomNewtonFirstLaw()


######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################


# Newtons Third Law


np.random.seed(42)
n_samples = 1000

F1 = np.random.uniform(-100, 100, size=n_samples) 
F2 = -F1  


def varyingnoiserandomNewtonThirdLaw():
    print("This is varying noise randomly: ")
    list_x = list(range(1,modeliterationsrandom))  
    y_results = list(map(noisy_third_law, list_x))  # This will store models
    print(y_results)  # Now prints actual model objects

    # save y to a file then later parse through it 
    with open("y_results_noise_Newtons_Third_Law.txt", "w") as file:
        file.write(",".join(map(str, y_results)))




def noisy_third_law(ran):

    X = F2.reshape(-1, 1) + (5*ran)
    y = F1 + (5*ran)



    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        populations=10,
        progress=True,
        model_selection="best",
        variable_names=["F2"]
    )

    model.fit(X, y)
    return model 

varyingnoiserandomNewtonThirdLaw()


######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################


# Newtons Third Law


np.random.seed(2)
n_samples = 100

t = np.linspace(0, 10, n_samples)               # Time from 0 to 10 seconds
omega = 2 * np.pi                               # Angular frequency (1 Hz)
x = np.cos(omega * t)   

def varyingnoiserandomSimpleHarmoicMotion():
    print("This is varying noise randomly: ")
    list_x = list(range(1,modeliterationsrandom))  
    y_results = list(map(noisy_simple_harmonic_motion, list_x))  # This will store models
    print(y_results)  # Now prints actual model objects

    # save y to a file then later parse through it 
    with open("y_results_noise_Simple_Harmonic_Motion.txt", "w") as file:
        file.write(",".join(map(str, y_results)))




def noisy_simple_harmonic_motion(ran):

    X = t.reshape(-1, 1) + (5*ran)
    y = x + (5*random.uniform(0,ran))

    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["cos", "sin"], 
        populations=10,
        progress=True,
        model_selection="best",
        variable_names=["t"]
    )

    model.fit(X, y)

    
varyingnoiserandomSimpleHarmoicMotion()

######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################


# Rutherford Scattering


np.random.seed(2)
n_samples = 100

theta = np.random.uniform(1e-2, np.pi - 1e-2, size=n_samples)  
k = 1.0  


def varyingnoiserandomRutherford():
    print("This is varying noise randomly: ")
    list_x = list(range(1,modeliterationsrandom))  
    y_results = list(map(noisy_rutherford, list_x))  # This will store models
    print(y_results)  # Now prints actual model objects

    # save y to a file then later parse through it 
    with open("y_results_noise_Rutherford.txt", "w") as file:
        file.write(",".join(map(str, y_results)))




def noisy_rutherford(ran):

    y = k / (np.sin(theta / 2) ** 4) + (5*ran)
    X = theta.reshape(-1, 1) + (5*random.uniform(0,ran))

    model = PySRRegressor(
        niterations=150,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sin", "cos"],
        populations=10,
        progress=True,
        model_selection="best",
        variable_names=["theta"]
    )

    model.fit(X, y)

    
varyingnoiserandomRutherford()

