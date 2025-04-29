import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


g = 9.81 
L1 = 1.0 
L2 = 1.0 
m1 = 1.0 
m2 = 1.0 

t_max = 20.0 
dt = 0.02    
t_span = [0, t_max]
t_eval = np.arange(0, t_max + dt, dt)

def deriv(t, S, L1, L2, m1, m2, g):
    theta1, omega1, theta2, omega2 = S
    d_theta = theta2 - theta1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(d_theta)**2
    den2 = (L2 / L1) * den1

    domega1_dt = (m2 * L1 * omega1**2 * np.sin(d_theta) * np.cos(d_theta) +
                  m2 * g * np.sin(theta2) * np.cos(d_theta) +
                  m2 * L2 * omega2**2 * np.sin(d_theta) -
                  (m1 + m2) * g * np.sin(theta1)) / den1

    domega2_dt = (-m2 * L2 * omega2**2 * np.sin(d_theta) * np.cos(d_theta) +
                  (m1 + m2) * g * np.sin(theta1) * np.cos(d_theta) -
                  (m1 + m2) * L1 * omega1**2 * np.sin(d_theta) -
                  (m1 + m2) * g * np.sin(theta2)) / den2

    dtheta1_dt = omega1
    dtheta2_dt = omega2

    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

def get_cartesian_coords(theta1, theta2, L1, L2):
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2

def plot_static_diagram(L1, L2, m1, m2, theta1_init, theta2_init):
    x1_init, y1_init, x2_init, y2_init = get_cartesian_coords(theta1_init, theta2_init, L1, L2)

    fig = go.Figure()

    fig.add_shape(type="line", x0=0, y0=0, x1=x1_init, y1=y1_init,
                  line=dict(color="black", width=3))
    fig.add_shape(type="line", x0=x1_init, y0=y1_init, x1=x2_init, y1=y2_init,
                  line=dict(color="black", width=3))

    bob_size1 = 8 + m1 * 4
    bob_size2 = 8 + m2 * 4
    fig.add_trace(go.Scatter(x=[0, x1_init, x2_init],
                             y=[0, y1_init, y2_init],
                             mode='markers',
                             marker=dict(color=['black', 'blue', 'red'],
                                         size=[6, bob_size1, bob_size2]),
                             name="Pivot & Bobs"))

    max_L = L1 + L2
    fig.update_layout(
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        xaxis_range=[-max_L * 1.1, max_L * 1.1],
        yaxis_range=[-max_L * 1.1, max_L * 1.1],
        yaxis_scaleanchor="x", 
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



initial_conditions = [
    {'theta1': np.pi * 0.6, 'omega1': 0.0, 'theta2': np.pi * 0.61, 'omega2': 0.0, 'name': 'Slightly Apart 1'},
    {'theta1': np.pi * 0.6, 'omega1': 0.0, 'theta2': np.pi * 0.615, 'omega2': 0.0, 'name': 'Slightly Apart 2'},
    {'theta1': np.pi * 0.8, 'omega1': 0.0, 'theta2': np.pi * 0.7, 'omega2': 0.0, 'name': 'Different Angles'},
    {'theta1': np.pi * 0.5, 'omega1': 0.0, 'theta2': np.pi * 1.0, 'omega2': 0.0, 'name': 'Wide Spread'},
    {'theta1': np.pi * 0.99, 'omega1': 0.0, 'theta2': np.pi * 1.0, 'omega2': 0.0, 'name': 'Nearly Vertical'},
]

path_colors = [
    'dodgerblue',   
    'crimson',     
    'darkviolet',  
    'mediumblue',  
    'firebrick',    
]
n_colors = len(path_colors) 

paths_fig = go.Figure()
all_paths_data = []

print("Simulating trajectories...")
for i, ic in enumerate(initial_conditions):
    print(f" Running simulation {i+1}/{len(initial_conditions)}: {ic['name']}")
    S0 = [ic['theta1'], ic['omega1'], ic['theta2'], ic['omega2']]

    sol = solve_ivp(deriv, t_span, S0, t_eval=t_eval, args=(L1, L2, m1, m2, g), method='RK45')

    if not sol.success:
        print(f"  WARNING: Simulation {i+1} failed: {sol.message}")
        continue

    theta1_sol = sol.y[0, :]
    theta2_sol = sol.y[2, :]

    x1, y1, x2, y2 = get_cartesian_coords(theta1_sol, theta2_sol, L1, L2)

    all_paths_data.append({'x': x2, 'y': y2, 'name': ic['name']})

    current_color = path_colors[i % n_colors] 

    paths_fig.add_trace(go.Scatter(
        x=x2,
        y=y2,
        mode='lines',
        name=ic['name'],
        line=dict(width=1.5, color=current_color)
    ))

print("Simulations complete. Generating plots...")

max_L = L1 + L2
paths_fig.update_layout(
    xaxis_title="x (m)",
    yaxis_title="y (m)",
    xaxis_range=[-max_L * 1.1, max_L * 1.1],
    yaxis_range=[-max_L * 1.1, max_L * 1.1],
    yaxis_scaleanchor="x",
    xaxis_constrain="domain",
    yaxis_constrain="domain",
    width=700,
    height=700,
    hovermode='closest',
    plot_bgcolor='white'
)
paths_fig.update_xaxes(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='grey')
paths_fig.update_yaxes(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='grey')



static_fig = plot_static_diagram(L1, L2, m1, m2,
                                 initial_conditions[0]['theta1'],
                                 initial_conditions[0]['theta2'])


print("Displaying static diagram: ")
static_fig.show()

print("Displaying path traces: ")
paths_fig.show()

print("Saving File: ")
pio.write_image(static_fig, "./dounle_pendulum_static", format='png')
pio.write_image(paths_fig, "./dounle_pendulum_path", format='png')

print("Script finished.")
