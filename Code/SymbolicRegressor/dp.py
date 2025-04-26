import numpy as np
from pysr import PySRRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.integrate import solve_ivp

# Parameters for the double pendulum system
m1, m2 = 1.0, 1.0  # masses of the pendulums
l1, l2 = 1.0, 1.0  # lengths of the pendulums
g = 9.81  # gravitational acceleration

# Define the equations of motion for the double pendulum
def derivatives(state, t):
    theta1, theta2, p1, p2 = state
    dtheta1_dt = (p1 / (m1 * l1**2)) - (m2 * g * np.sin(theta1) * np.cos(theta1 - theta2)) / (m1 * l1)
    dtheta2_dt = (p2 / (m2 * l2**2)) - (m2 * g * np.sin(theta2) * np.cos(theta2 - theta1)) / (m2 * l2)
    dp1_dt = -g * (2 * m1 + m2) * np.sin(theta1)
    dp2_dt = -g * m2 * np.sin(theta2)
    
    return [dtheta1_dt, dtheta2_dt, dp1_dt, dp2_dt]

# Set initial conditions
initial_state = [np.pi / 2, np.pi / 2, 0.0, 0.0]  # Initial angles and angular velocities
t = np.linspace(0, 10, 1000)  # Time grid

# Integrate the system using Runge-Kutta
sol = solve_ivp(derivatives, [0, 10], initial_state, t_eval=t)

# Extract data from the solution
theta1_data = sol.y[0]
theta2_data = sol.y[1]
theta1_dot_data = sol.y[2]
theta2_dot_data = sol.y[3]


# Prepare the data
data = np.column_stack([theta1_data, theta2_data, theta1_dot_data, theta2_dot_data])
# Target values for the regression can be the accelerations (derivatives of velocities)
# We will approximate the accelerations for theta1 and theta2 using finite differences
theta1_ddot_data = np.gradient(theta1_dot_data, t)  # Approximate second derivative
theta2_ddot_data = np.gradient(theta2_dot_data, t)

# PySR requires the data to be in a DataFrame format

df = pd.DataFrame(data, columns=["theta1", "theta2", "theta1_dot", "theta2_dot"])

# Add the target values (accelerations) to the DataFrame
df["theta1_ddot"] = theta1_ddot_data
df["theta2_ddot"] = theta2_ddot_data

# Create a PySR regressor and fit the model
model = PySRRegressor(
    niterations=1000,  # Number of iterations (adjust based on computational resources)
    binary_operators=["+", "-", "*", "**"],  # Allowed operators
    unary_operators=["sin", "cos", "exp", "log"],  # Allowed functions
    multithreading=True,  # Enable multithreading for speed
)

# Fit the model to the data (we are using theta1, theta2, theta1_dot, and theta2_dot to predict theta1_ddot and theta2_ddot)
model.fit(df[["theta1", "theta2", "theta1_dot", "theta2_dot"]], df[["theta1_ddot", "theta2_ddot"]])

# Print the best symbolic equation found by the model
print(model)

