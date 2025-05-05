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


modeliterationsrandom = 4; 

np.random.seed(0)
n_samples = 100
t = np.linspace(0, 10, n_samples) 
omega = 2 * np.pi  
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

    X = t.reshape(-1, 1) + (5*random.uniform(0,ran))
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
    return model 

    
varyingnoiserandomSimpleHarmoicMotion()


