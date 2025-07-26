# Geodesic Mesh Visualizer

A Dash-based web application to visualize a 3D mesh and compute the geodesic (surface) distance between two points on the mesh. The app allows users to interactively select points on the mesh, snap them to the nearest vertices, and compute the shortest path along the mesh surface.

## Features
- Load and visualize 3D meshes (OBJ format).
- Interactively select two points on the mesh.
- Compute the geodesic distance using Dijkstra's algorithm on a graph built from mesh edges.
- Visualize the mesh, selected points, and geodesic path using Plotly.

## Prerequisites
- Python 3.8 or higher
- A 3D mesh file in OBJ format (e.g., `preprocessed.obj`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/geodesic-mesh-visualizer.git
   cd geodesic-mesh-visualizer