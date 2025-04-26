import os
import json
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

def export_to_nerfstudio(pose_json_path, object_xyz_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # === 1. Load poses ===
    with open(pose_json_path, 'r') as f:
        poses = json.load(f)

    frames = []
    for idx, pose in enumerate(poses):
        position = np.array(pose["position"])
        direction = np.array(pose["direction"])
        uncertainty = pose["uncertainty"]

        # Build camera-to-world matrix
        forward = direction / np.linalg.norm(direction)
        up = np.array([0, 1, 0])  # simple up vector
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = position

        # Color coding: green (low uncertainty) to red (high)
        uncertainty = np.clip(uncertainty, 0, 1)
        color = [
            int(255 * uncertainty),
            int(255 * (1 - uncertainty)),
            0
        ]

        frame = {
            "file_path": f"./dummy_images/{idx:04d}.png",  # dummy since we only visualize poses
            "transform_matrix": c2w.tolist(),
            "uncertainty_color": color,
            "fl_x": pose["FoVx"], "fl_y": pose["FoVy"],
            "cx": 0.5, "cy": 0.5,
            "w": 800, "h": 800
        }
        frames.append(frame)

    transforms = {
        "frames": frames,
        "camera_model": "OPENCV",
        "w": 800,
        "h": 800,
        "fl_x": poses[0]["FoVx"],
        "fl_y": poses[0]["FoVy"],
        "cx": 0.5,
        "cy": 0.5
    }

    with open(os.path.join(output_folder, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=4)

    print(f"Saved Nerfstudio transforms.json to {output_folder}")

    # === 2. Save object points as PLY ===
    object_xyz = np.load(object_xyz_path)

    vertex = np.array(
        [tuple(pt) for pt in object_xyz],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(os.path.join(output_folder, 'object.ply'))

    print(f"Saved object.ply to {output_folder}")

# === Usage ===
export_to_nerfstudio(
    pose_json_path="vis_output/nbv_custom_poses.json",
    object_xyz_path="vis_output/object_xyz.npy",
    output_folder="my_vis"
)
