import json
import numpy as np
from scene.cameras import Camera

def load_cameras_from_pose_file(path_to_json, resolution, device):
    with open(path_to_json, 'r') as f:
        pose_dicts = json.load(f)

    cameras = []
    for idx, pose in enumerate(pose_dicts):
        R = np.array(pose['R'], dtype=np.float32)
        T = np.array(pose['T'], dtype=np.float32)
        FoVx = pose['FoVx']
        FoVy = pose['FoVy']

        cam = Camera(
            colmap_id=idx,
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            image=None,  # <-- This is correct
            image_name=f"pose_{idx:03d}",
            uid=idx,
            data_device=device
        )
        cameras.append(cam)
    return cameras

