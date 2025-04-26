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

    # Now create data for poses + arrows
    arrow_length = 0.3

    line_starts = []
    line_ends = []

    for pos, dir in zip(positions, directions):
        start = pos
        end = pos + arrow_length * np.array(dir)
        line_starts.append(start)
        line_ends.append(end)

    # Stack points for lines
    lines = []
    for s, e in zip(line_starts, line_ends):
        lines.append(s)
        lines.append(e)
        lines.append([np.nan, np.nan, np.nan])  # break in line

    lines = np.array(lines)

    # Try to load reconstructed object
    try:
        object_points = np.load("vis_output/object_xyz.npy")
        if object_points.shape[0] > 5000:
            idx = np.random.choice(object_points.shape[0], 5000, replace=False)
            object_points = object_points[idx]
    except Exception as e:
        print("[Visualizer] Could not load object point cloud:", e)
        object_points = np.zeros((0, 3))  # fallback

    # Final scatter points
    all_points = np.vstack([positions, object_points])

    # Colors: poses are colored, object is gray
    pose_colors = colors
    object_colors = np.tile(np.array([[180, 180, 180]]), (object_points.shape[0], 1))
    all_colors = np.vstack([pose_colors, object_colors])

    vis.scatter(
        X=all_points,
        opts=dict(
            markersize=5,
            markercolor=all_colors.astype(np.uint8),
            title="Camera Poses + Object",
            xlabel="X", ylabel="Y", zlabel="Z"
        ),
        win="combined"
    )

    # Draw directions as lines
    if len(lines) > 0:
        vis.scatter(
            X=lines,
            opts=dict(
                markersize=0,
                linecolor=np.array([[0, 0, 255]]),
                mode="lines",
                title="Camera Directions",
                xlabel="X", ylabel="Y", zlabel="Z"
            ),
            win="camera_dirs"
        )
