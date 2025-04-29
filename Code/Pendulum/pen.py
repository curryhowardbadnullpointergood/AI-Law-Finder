import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Physical Parameters ---
g = 9.81  # Acceleration due to gravity (m/s^2)
L1 = 1.0  # Length of the first pendulum arm (m)
L2 = 1.0  # Length of the second pendulum arm (m)
m1 = 1.0  # Mass of the first bob (kg)
m2 = 1.0  # Mass of the second bob (kg)

# --- Simulation Parameters ---
t_max = 20.0  # Maximum simulation time (s)
dt = 0.02     # Time step for output (s)
t_span = [0, t_max]
t_eval = np.arange(0, t_max + dt, dt)

# --- Equations of Motion ---
# State vector S = [theta1, omega1, theta2, omega2]
# theta1, theta2: angles of the arms from the vertical
# omega1, omega2: angular velocities
def deriv(t, S, L1, L2, m1, m2, g):
    theta1, omega1, theta2, omega2 = S

    # Intermediate delta theta
    d_theta = theta2 - theta1

    # Denominators (avoid recalculation)
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(d_theta)**2
    den2 = (L2 / L1) * den1

    # Derivatives of omegas (accelerations)
    domega1_dt = (m2 * L1 * omega1**2 * np.sin(d_theta) * np.cos(d_theta) +
                  m2 * g * np.sin(theta2) * np.cos(d_theta) +
                  m2 * L2 * omega2**2 * np.sin(d_theta) -
                  (m1 + m2) * g * np.sin(theta1)) / den1

    domega2_dt = (-m2 * L2 * omega2**2 * np.sin(d_theta) * np.cos(d_theta) +
                  (m1 + m2) * g * np.sin(theta1) * np.cos(d_theta) -
                  (m1 + m2) * L1 * omega1**2 * np.sin(d_theta) -
                  (m1 + m2) * g * np.sin(theta2)) / den2

    # Derivatives of state vector
    dtheta1_dt = omega1
    dtheta2_dt = omega2

    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

# --- Coordinate Conversion ---
def get_cartesian_coords(theta1, theta2, L1, L2):
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2

# --- Static Diagram Function ---
def plot_static_diagram(L1, L2, m1, m2, theta1_init, theta2_init):
    # Calculate initial positions for the diagram
    x1_init, y1_init, x2_init, y2_init = get_cartesian_coords(theta1_init, theta2_init, L1, L2)

    fig = go.Figure()

    # Add Rods
    fig.add_shape(type="line", x0=0, y0=0, x1=x1_init, y1=y1_init,
                  line=dict(color="black", width=3))
    fig.add_shape(type="line", x0=x1_init, y0=y1_init, x1=x2_init, y1=y2_init,
                  line=dict(color="black", width=3))

    # Add Pivot and Bobs
    bob_size1 = 8 + m1 * 4 # Adjust size based on mass
    bob_size2 = 8 + m2 * 4
    fig.add_trace(go.Scatter(x=[0, x1_init, x2_init],
                             y=[0, y1_init, y2_init],
                             mode='markers',
                             marker=dict(color=['black', 'blue', 'red'],
                                         size=[6, bob_size1, bob_size2]),
                             name="Pivot & Bobs"))

    # Add Angle Arcs (optional, can be complex to get right visually)
    # Example for theta1 (simplified)
    # r_arc = L1 * 0.2
    # arc_theta = np.linspace(-np.pi/2, theta1_init - np.pi/2, 20)
    # x_arc = r_arc * np.cos(arc_theta)
    # y_arc = r_arc * np.sin(arc_theta)
    # fig.add_trace(go.Scatter(x=x_arc, y=y_arc, mode='lines', line=dict(color='grey', dash='dot'), name='θ1'))

    # Set layout
    max_L = L1 + L2
    fig.update_layout(
        title="Static Diagram of Double Pendulum",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        xaxis_range=[-max_L * 1.1, max_L * 1.1],
        yaxis_range=[-max_L * 1.1, max_L * 1.1],
        yaxis_scaleanchor="x", # Ensures correct aspect ratio
        xaxis_constrain="domain",
        yaxis_constrain="domain",
        width=600,
        height=600,
        showlegend=True,
        plot_bgcolor='white'
    )
    fig.update_xaxes(showgrid=False, zeroline=True, zerolinecolor='lightgrey')
    fig.update_yaxes(showgrid=False, zeroline=True, zerolinecolor='lightgrey')

    return fig

# --- Simulation and Path Plotting ---

# Define several initial conditions (angles in radians, angular velocities = 0)
initial_conditions = [
    {'theta1': np.pi * 0.6, 'omega1': 0.0, 'theta2': np.pi * 0.61, 'omega2': 0.0, 'name': 'Slightly Apart 1'},
    {'theta1': np.pi * 0.6, 'omega1': 0.0, 'theta2': np.pi * 0.615, 'omega2': 0.0, 'name': 'Slightly Apart 2'},
    {'theta1': np.pi * 0.8, 'omega1': 0.0, 'theta2': np.pi * 0.7, 'omega2': 0.0, 'name': 'Different Angles'},
    {'theta1': np.pi * 0.5, 'omega1': 0.0, 'theta2': np.pi * 1.0, 'omega2': 0.0, 'name': 'Wide Spread'},
    {'theta1': np.pi * 0.99, 'omega1': 0.0, 'theta2': np.pi * 1.0, 'omega2': 0.0, 'name': 'Nearly Vertical'},
]

paths_fig = go.Figure()
all_paths_data = []

print("Simulating trajectories...")
for i, ic in enumerate(initial_conditions):
    print(f" Running simulation {i+1}/{len(initial_conditions)}: {ic['name']}")
    S0 = [ic['theta1'], ic['omega1'], ic['theta2'], ic['omega2']]

    # Solve the ODE
    sol = solve_ivp(deriv, t_span, S0, t_eval=t_eval, args=(L1, L2, m1, m2, g), method='RK45') # Could use LSODA for potentially stiffer cases

    # Check if integration was successful
    if not sol.success:
        print(f"  WARNING: Simulation {i+1} failed: {sol.message}")
        continue

    # Get angles from solution
    theta1_sol = sol.y[0, :]
    theta2_sol = sol.y[2, :]

    # Convert to Cartesian coordinates
    x1, y1, x2, y2 = get_cartesian_coords(theta1_sol, theta2_sol, L1, L2)

    # Store data for plotting
    all_paths_data.append({'x': x2, 'y': y2, 'name': ic['name']})

    # Add path trace to the figure
    paths_fig.add_trace(go.Scatter(
        x=x2,
        y=y2,
        mode='lines',
        name=ic['name'],
        line=dict(width=1.5) # Slightly thicker lines
    ))

print("Simulations complete. Generating plots...")

# Configure paths plot layout
max_L = L1 + L2
paths_fig.update_layout(
    title="Paths Traced by the Second Bob (Various Initial Conditions)",
    xaxis_title="x (m)",
    yaxis_title="y (m)",
    xaxis_range=[-max_L * 1.1, max_L * 1.1],
    yaxis_range=[-max_L * 1.1, max_L * 1.1],
    yaxis_scaleanchor="x", # Crucial for correct visual representation
    xaxis_constrain="domain",
    yaxis_constrain="domain",
    width=700,
    height=700,
    legend_title="Initial Condition Set",
    hovermode='closest', # Show hover info for nearest point
    plot_bgcolor='white'
)
paths_fig.update_xaxes(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='grey')
paths_fig.update_yaxes(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='grey')


# --- Display Plots ---

# Show static diagram (using first initial condition for example pose)
static_fig = plot_static_diagram(L1, L2, m1, m2,
                                 initial_conditions[0]['theta1'],
                                 initial_conditions[0]['theta2'])
print("Displaying static diagram...")
static_fig.show()

# Show paths plot
print("Displaying path traces...")
paths_fig.show()

print("Script finished.")
