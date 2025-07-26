import numpy as np
import trimesh
import networkx as nx
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State

# ----------------------------------
# Utility Functions
# ----------------------------------

def load_mesh(filename):
    """
    Load and process the mesh.
    """
    mesh = trimesh.load(filename)
    mesh.process()  # Compute necessary attributes (edges, normals, etc.)
    print(f"✅ Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    return mesh

def build_full_graph_from_mesh(mesh):
    """
    Build a full graph from all edges in the mesh.
    Every vertex becomes a node and every edge (from each face) is added with its Euclidean length.
    This graph is used to compute the geodesic path (shortest path along the mesh surface).
    """
    G = nx.Graph()
    vertices = mesh.vertices
    for i in range(len(vertices)):
        G.add_node(i)
    for face in mesh.faces:
        for j in range(3):
            u = face[j]
            v = face[(j+1) % 3]
            dist = float(np.linalg.norm(vertices[u] - vertices[v]))
            # Add edge in both directions (Graph is undirected)
            G.add_edge(u, v, weight=dist)
    return G

def compute_geodesic_distance(mesh, start_xyz, end_xyz, G):
    """
    Compute the geodesic (surface) distance between start_xyz and end_xyz using the full mesh graph.
    
    Returns:
      (distance, path_list) where:
         - distance is the geodesic distance (shortest path along mesh edges),
         - path_list is the list of [x,y,z] coordinates along the computed path.
    If no path is found, returns (None, None).
    """
    vertices = mesh.vertices
    # Snap the given points to the nearest vertex (using all vertices)
    start_index = int(np.argmin(np.linalg.norm(vertices - start_xyz, axis=1)))
    end_index   = int(np.argmin(np.linalg.norm(vertices - end_xyz, axis=1)))
    
    try:
        shortest_path_indices = nx.shortest_path(G, source=start_index, target=end_index, weight='weight')
        path_vertices = vertices[shortest_path_indices]
        distance = float(np.sum(np.linalg.norm(np.diff(path_vertices, axis=0), axis=1)))
        path_list = [[float(c) for c in pt] for pt in path_vertices.tolist()]
        return distance, path_list
    except nx.NetworkXNoPath:
        return None, None

def create_figure(mesh, selected_points=None, path=None):
    """
    Create a 3D Plotly figure showing:
      - The mesh (gray markers)
      - The selected points (red markers with labels)
      - The computed geodesic path (blue line)
    """
    fig = go.Figure()
    
    # Plot mesh vertices (as gray markers)
    fig.add_trace(go.Scatter3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        mode='markers',
        marker=dict(size=2, color='gray', opacity=0.5),
        name='Mesh',
        customdata=mesh.vertices.tolist()
    ))
    
    # Plot selected points
    if selected_points:
        fig.add_trace(go.Scatter3d(
            x=[pt[0] for pt in selected_points],
            y=[pt[1] for pt in selected_points],
            z=[pt[2] for pt in selected_points],
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=[f"P{i+1}" for i in range(len(selected_points))],
            textposition='top center',
            name='Selected Points'
        ))
    
    # Plot computed geodesic path
    if path:
        fig.add_trace(go.Scatter3d(
            x=[pt[0] for pt in path],
            y=[pt[1] for pt in path],
            z=[pt[2] for pt in path],
            mode='lines',
            line=dict(width=4, color='blue'),
            name='Geodesic Path'
        ))
    
    # Adjust axes based on mesh bounds
    x_min, x_max = np.min(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 0])
    y_min, y_max = np.min(mesh.vertices[:, 1]), np.max(mesh.vertices[:, 1])
    z_min, z_max = np.min(mesh.vertices[:, 2]), np.max(mesh.vertices[:, 2])
    fig.update_layout(
        title="Geodesic Distance on Mesh Surface",
        scene=dict(
            xaxis=dict(range=[x_min, x_max], title="X"),
            yaxis=dict(range=[y_min, y_max], title="Y"),
            zaxis=dict(range=[z_min, z_max], title="Z"),
            aspectmode='data'
        )
    )
    return fig

# ----------------------------------
# Main Code and Dash App Setup
# ----------------------------------

mesh = load_mesh("preprocessed.obj")
# Build the full graph from mesh edges (all edges are included)
graph_full = build_full_graph_from_mesh(mesh)

app = Dash(__name__)
app.layout = html.Div([
    html.H1("Measure Geodesic Distance on Mesh", style={'textAlign': 'center'}),
    dcc.Graph(id='3d-graph', figure=create_figure(mesh, []), style={'height': '70vh'}),
    html.Div("Click on the mesh to select 2 points (they will be snapped to the surface).", 
             style={'marginTop': 20, 'fontSize': 18, 'textAlign': 'center'}),
    html.Button("Compute Distance", id='compute-btn', n_clicks=0, style={'marginTop': 10}),
    html.Div(id='measurement-text', style={
         'marginTop': 20, 'fontSize': 22, 'fontWeight': 'bold', 'color': 'blue', 'textAlign': 'center'
    }),
    dcc.Store(id='selected-points', data=[])
])

@app.callback(
    [Output('3d-graph', 'figure'),
     Output('measurement-text', 'children'),
     Output('selected-points', 'data')],
    [Input('3d-graph', 'clickData'),
     Input('compute-btn', 'n_clicks')],
    State('selected-points', 'data')
)
def update_app(clickData, n_clicks, stored_points):
    # Make sure stored_points is a list.
    if stored_points is None:
        stored_points = []
    else:
        stored_points = list(stored_points)
    
    # When the graph is clicked and fewer than 2 points have been selected, snap the click to the nearest vertex.
    if clickData and len(stored_points) < 2:
        click_pt = clickData['points'][0]
        # Use the click coordinates from either 'z' or 'customdata'
        if 'z' in click_pt and click_pt['z'] is not None:
            click_coords = np.array([float(click_pt['x']), float(click_pt['y']), float(click_pt['z'])])
        elif 'customdata' in click_pt and click_pt['customdata']:
            click_coords = np.array([float(click_pt['customdata'][0]),
                                       float(click_pt['customdata'][1]),
                                       float(click_pt['customdata'][2])])
        else:
            click_coords = np.array([float(click_pt['x']), float(click_pt['y']), 0.0])
        # Snap to the nearest vertex (using all vertices)
        vertices = mesh.vertices
        distances = np.linalg.norm(vertices - click_coords, axis=1)
        closest_index = int(np.argmin(distances))
        new_pt = mesh.vertices[closest_index].tolist()
        stored_points.append(new_pt)
    
    measurement_msg = ""
    path = None
    if len(stored_points) < 2:
        measurement_msg = f"Selected {len(stored_points)} point(s). Please select 2 points."
    elif len(stored_points) == 2 and n_clicks > 0:
        start_pt = np.array(stored_points[0])
        end_pt = np.array(stored_points[1])
        d, path = compute_geodesic_distance(mesh, start_pt, end_pt, G=graph_full)
        if d is None:
            measurement_msg = "⚠️ Could not compute a geodesic path between the points."
        else:
            measurement_msg = f"Geodesic Distance: {d:.3f} units"
    else:
        if len(stored_points) == 2:
            measurement_msg = "Click 'Compute Distance' to measure."
    
    fig = create_figure(mesh, selected_points=stored_points, path=path)
    return fig, measurement_msg, stored_points

if __name__ == '__main__':
    app.run(debug=True, port=8051)
