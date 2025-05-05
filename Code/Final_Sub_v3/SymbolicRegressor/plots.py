import plotly.express as px
import pandas as pd

def plot_pareto_frontier_plotly(pareto_points, title="Modern Pareto Frontier"):
    df = pd.DataFrame({
        'Complexity': [pt[0] for pt in pareto_points],
        'Loss': [pt[1] for pt in pareto_points],
        'Expression': [str(pt[2]) for pt in pareto_points]
    })

    fig = px.scatter(df, x='Complexity', y='Loss', text='Expression', title=title)
    fig.update_traces(marker=dict(size=10, color='red'), textposition='top center')
    fig.update_layout(
        plot_bgcolor='lightgrey',
        paper_bgcolor='white',
        font=dict(family="Arial", size=14),
        title_x=0.5
    )

    return fig 



def plot_gradient_importance(gradients):
    mean_importance = np.mean(np.abs(gradients), axis=0)
    labels = [f"Π{i+1}" for i in range(len(mean_importance))]

    fig = px.bar(x=labels, y=mean_importance, 
                 labels={'x': 'Dimensionless Variable', 'y': 'Mean |∂f/∂Πi|'},
                 title="Variable Importance from Gradient Magnitudes")
    fig.update_traces(marker_color='royalblue')
    fig.update_layout(template='plotly_white')
    fig.show()



def plot_dependency_graph_plotly(gradients):
    mean_grads = np.mean(np.abs(gradients), axis=0)
    G = nx.DiGraph()
    for i, weight in enumerate(mean_grads):
        G.add_edge(f"Π{i+1}", "Output", weight=weight)

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y, edge_weights = [], [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_weights.append(G.edges[edge]['weight'])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='lightblue'),
        hoverinfo='none',
        mode='lines')

    node_x, node_y, labels = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=labels,
        textposition="top center",
        marker=dict(size=40, color='lightgreen', line=dict(width=2, color='black')),
        hoverinfo='text')

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Model Dependency Graph (Gradient-based)',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        template='plotly_white'
                    ))
    fig.show()


