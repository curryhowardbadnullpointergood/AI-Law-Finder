import numpy as np
import plotly.graph_objects as go
import os # Import os module to handle file paths

# --- Simulation Parameters ---
g = 9.81  # Acceleration due to gravity (m/s^2)
L = 1.0   # Length of the pendulum (m)
dt = 0.05 # Time step (s) - smaller is more accurate but slower
total_time = 10.0 # Total simulation time (s)

# --- Initial Conditions ---
# Angle (theta) is measured from the downward vertical (0 rad)
initial_theta = np.pi / 2  # Starting angle (e.g., pi/2 = 90 degrees)
initial_omega = 0.0      # Starting angular velocity (rad/s)

# --- Simulation Setup ---
n_steps = int(total_time / dt)
time_points = np.linspace(0, total_time, n_steps + 1)
theta_values = np.zeros(n_steps + 1)
omega_values = np.zeros(n_steps + 1)

# Set initial values
theta_values[0] = initial_theta
omega_values[0] = initial_omega

# --- Simulation Loop (Euler-Cromer method) ---
for i in range(n_steps):
    # Calculate angular acceleration
    alpha = -(g / L) * np.sin(theta_values[i])

    # Update angular velocity
    omega_values[i+1] = omega_values[i] + alpha * dt

    # Update angle using the *new* velocity
    theta_values[i+1] = theta_values[i] + omega_values[i+1] * dt

    # Optional: Keep angle within [-pi, pi] for cleaner plotting if needed,
    # though sin() handles larger angles correctly.
    # theta_values[i+1] = (theta_values[i+1] + np.pi) % (2 * np.pi) - np.pi

# --- Plotting with Plotly ---
fig = go.Figure()

# Add the trajectory trace
fig.add_trace(go.Scatter(
    x=time_points,
    y=theta_values,
    mode='lines+markers', # Show discrete steps and connecting line
    name=f'Pendulum Path (L={L}m, θ₀={initial_theta:.2f} rad)',
    marker=dict(size=4) # Smaller markers
))

# Update layout for clarity
fig.update_layout(
    title=f'Simple Pendulum Motion (Euler Method)<br>L={L}m, θ₀={initial_theta:.2f} rad, ω₀={initial_omega:.2f} rad/s',
    xaxis_title='Time (s)',
    yaxis_title='Angle θ (radians)',
    yaxis=dict(range=[min(theta_values)*1.1, max(theta_values)*1.1]), # Adjust y-axis range
    hovermode="x unified",
    template="plotly_white" # Use a clean template
)

# --- Display the plot ---
print(f"Simulated {n_steps} steps over {total_time} seconds.")
fig.show()

# --- Save the plot as an image ---
# Define the output filename
output_filename = "pendulum_simulation.png" # You can change the format (e.g., .jpg, .svg, .pdf)

# Check if the kaleido package is installed (required for static image export)
try:
    # Save the figure
    fig.write_image(output_filename)
    print(f"Plot saved successfully as '{output_filename}'")
except ValueError as e:
    if "kaleido" in str(e):
        print("\nError saving plot: The 'kaleido' package is required for static image export.")
        print("Please install it using: pip install -U kaleido")
    else:
        print(f"\nAn error occurred while saving the plot: {e}")
except Exception as e:
     print(f"\nAn unexpected error occurred while saving the plot: {e}")


