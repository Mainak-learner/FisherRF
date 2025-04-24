# vis/launch_viewer.py
import subprocess
import time
import json
import numpy as np
from visdom import Visdom
from pyngrok import ngrok

def launch_viewer_from_json(json_path, ngrok_token="YOUR_NGROK_TOKEN"):
    print("[Visualizer] Starting Visdom server...")

    subprocess.Popen(["python3", "-m", "visdom.server", "-port", "8097"])
    time.sleep(2)

    ngrok.set_auth_token(ngrok_token)
    tunnel = ngrok.connect(8097)
    visdom_url = tunnel.public_url  # string!
    print(f"[Visualizer] 🔗 View here: {visdom_url}")

    # Strip 'https://' but pass only the hostname to Visdom
    vis = Visdom(server=visdom_url.replace("https://", ""))
    with open(json_path, 'r') as f:
        poses = json.load(f)

    positions = np.array([p["position"] for p in poses])
    uncertainties = np.array([p["uncertainty"] for p in poses])
    colors = (1 - uncertainties)[:, None] * np.array([[0,255,0]]) + uncertainties[:, None] * np.array([[255,0,0]])

    vis.scatter(
        X=positions,
        opts=dict(
            markersize=6,
            markercolor=colors.astype(np.uint8),
            title="Generated Poses with Uncertainty",
            xlabel="X", ylabel="Y", zlabel="Z"
        )
    )

    try:
        object_points = np.load("vis_output/object_xyz.npy")
        vis.scatter(
            X=object_points,
            opts=dict(
                markersize=2,
                markercolor=np.tile(np.array([[180, 180, 180]]), (object_points.shape[0], 1)),  # light gray
                title="Reconstructed Object"
            ),
            win="object_points"
        )
    except Exception as e:
        print("[Visualizer] Could not load object point cloud:", e)