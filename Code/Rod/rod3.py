import numpy as np
import plotly.graph_objects as go

rod_length = 10.0
rod_radius = 0.5
n_length = 50  
n_circumference = 30 
max_temp = 100.0 
min_temp = 20.0 
colormap_name = 'Hot' 

x_lin = np.linspace(0, rod_length, n_length)
theta = np.linspace(0, 2 * np.pi, n_circumference)

x_grid, theta_grid = np.meshgrid(x_lin, theta)

X = x_grid
Y = rod_radius * np.cos(theta_grid)
Z = rod_radius * np.sin(theta_grid)

Temperature = max_temp - (X / rod_length) * (max_temp - min_temp)

fig = go.Figure(data=[
    go.Surface(
        x=X,
        y=Y,
        z=Z,
        surfacecolor=Temperature, 
        colorscale=colormap_name, 
        cmin=min_temp,            
        cmax=max_temp,            
        colorbar=dict(            
            title='Temperature (°C)',
            thickness=20,
            ticklen=5,
            len=0.75 
        ),
        showscale=True 
    )
])

fig.update_layout(
    title='Temperature Distribution on a Heated Rod (Plotly)',
    scene=dict(
        xaxis_title='Length (x)',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectratio=dict(x=5, y=1, z=1),
        aspectmode='manual', 
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.8)
        )
    ),
    margin=dict(l=0, r=0, b=0, t=40) 
)


try:
    fig.write_image("heated_rod_plotly.png", scale=2)
    print("Saved image as 'heated_rod_plotly.png'")
except ValueError as e:
     print(f"Error saving PNG: {e}. Make sure 'kaleido' is installed (`pip install -U kaleido`)")



fig.show()
