# vis/launch_viewer.py

import subprocess
import time
import json
import numpy as np
import logging
from pyngrok import ngrok
from urllib.parse import urlparse
from visdom import Visdom

def launch_viewer_from_json(json_path, ngrok_token):
    print("[Visualizer] Starting Visdom server...")

    # Hide Visdom warnings
    logging.getLogger('visdom').setLevel(logging.ERROR)

    subprocess.Popen(["python3", "-m", "visdom.server", "-port", "8097"])
    time.sleep(5)

    ngrok.set_auth_token(ngrok_token)
    tunnel = ngrok.connect(8097, bind_tls=True)
    print(f"[Visualizer] 🔗 View here: {tunnel.public_url}")

    parsed = urlparse(tunnel.public_url)
    visdom_host = parsed.hostname

    vis = Visdom(server=visdom_host, port=80, use_incoming_socket=False)

    # Load pose data
    with open(json_path, 'r') as f:
        poses = json.load(f)

    positions = np.array([p["position"] for p in poses])
    uncertainties = np.array([p["uncertainty"] for p in poses])
    directions = np.array([p["direction"] for p in poses])
    
    colors = (1 - uncertainties)[:, None] * np.array([[0,255,0]]) + uncertainties[:, None] * np.array([[255,0,0]])

    # Plot camera centers
    vis.scatter(
        X=positions,
        opts=dict(
            markersize=6,
            markercolor=colors.astype(np.uint8),
            title="Generated Poses",
            xlabel="X", ylabel="Y", zlabel="Z"
        ),
        win="cameras"
    )

    # Plot view directions as small lines ("mini-cones")
    line_segments = []
    line_colors = []

    arrow_length = 0.1  # Small arrows

    for pos, dir in zip(positions, directions):
        start = pos
        end = pos + arrow_length * np.array(dir)
        line_segments.append(np.vstack((start, end)))
        line_colors.append(np.array([0, 0, 255]))  # Blue arrows

    if len(line_segments) > 0:
        lines = np.vstack(line_segments)
        line_color_array = np.vstack([line_colors for _ in range(len(line_segments))])
        
        vis.line(
            X=lines[:, [0]], Y=lines[:, [1]], opts=dict(
                markers=False,
                linecolor=line_color_array.tolist(),
                xlabel='X', ylabel='Y',
                title="Camera Directions (XZ plane)"
            ),
            win="camera_dirs"
        )

    # Plot reconstructed object points
    try:
        object_points = np.load("vis_output/object_xyz.npy")

        # Optional: downsample if too big
        if object_points.shape[0] > 5000:
            idx = np.random.choice(object_points.shape[0], 5000, replace=False)
            object_points = object_points[idx]

        vis.scatter(
            X=object_points,
            opts=dict(
                markersize=2,
                markercolor=np.tile(np.array([[180, 180, 180]]), (object_points.shape[0], 1)),  # light gray
                title="Reconstructed Object (Point Cloud)"
            ),
            win="object_points"
        )
    except Exception as e:
        print("[Visualizer] Could not load object point cloud:", e)
