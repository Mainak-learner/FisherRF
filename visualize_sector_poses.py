import numpy as np
import torch
from jupyter_dash import JupyterDash
import plotly.graph_objects as go
from dash import dcc, html, Input, Output

# ---------- CONFIGURATION ----------
# Load from .npy or hardcode your tensors here
all_centers = np.load("all_centers.npy")         # Shape: (N, 3)
object_center = np.load("object_center.npy")     # Shape: (3,)
middle_indices = np.load("middle_circle_indices.npy")  # Optional: just for coloring

# ---------- NORMALIZATION ----------
camera_positions = all_centers
object_center_np = object_center

# Compute direction vectors
camera_directions = object_center_np[None, :] - camera_positions
camera_directions /= np.linalg.norm(camera_directions, axis=1, keepdims=True)

# ---------- FRUSTUM VISUALIZATION ----------
frustum_traces = []
frustum_scale = 0.2
for pos, dir in zip(camera_positions, camera_directions):
    fwd = dir
    up = np.array([0, 0, 1]) if abs(np.dot(fwd, [0, 0, 1])) < 0.9 else np.array([1, 0, 0])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right); up /= np.linalg.norm(up)

    base = [pos + frustum_scale * (fwd + right * xr + up * yr)
            for xr, yr in [(1, 1), (-1, 1), (-1, -1), (1, -1)]]
    edges = [(pos, b) for b in base] + [(base[i], base[(i + 1) % 4]) for i in range(4)]

    x, y, z = [], [], []
    for s, e in edges:
        x += [s[0], e[0], None]
        y += [s[1], e[1], None]
        z += [s[2], e[2], None]

    frustum_traces.append(go.Scatter3d(
        x=x, y=y, z=z, mode='lines',
        line=dict(color='red', width=1.2),
        showlegend=False
    ))

# ---------- MAIN PLOT ----------
pose_colors = ['blue'] * len(camera_positions)
if 'middle_indices' in locals():
    for idx in middle_indices:
        pose_colors[idx] = 'orange'

fig = go.Figure(data=[
    go.Scatter3d(
        x=camera_positions[:, 0], y=camera_positions[:, 1], z=camera_positions[:, 2],
        mode='markers',
        marker=dict(size=4, color=pose_colors),
        name='Generated Poses'
    ),
    go.Scatter3d(
        x=[object_center_np[0]], y=[object_center_np[1]], z=[object_center_np[2]],
        mode='markers+text',
        marker=dict(size=6, color='green'),
        text=['Object Center'], name='Center'
    )
] + frustum_traces)

fig.update_layout(scene=dict(aspectmode='data'), height=700)

# ---------- DASH APP ----------
app = JupyterDash(__name__)
app.layout = html.Div([
    html.H2("Sector-based Pose Sampling Visualization"),
    dcc.Graph(id='plot-3d', figure=fig)
])

# Run with: python visualize_sector_poses.py
if __name__ == "__main__":
    app.run_server(debug=False, port=8050)
