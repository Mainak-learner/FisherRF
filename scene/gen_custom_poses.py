import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from render_uncertainty_from_poses import capture

def extract_object_center_from_gaussians(scene):
    means = list(capture(scene.gaussians)[1:7])
    return means[0].mean(0).detach().cpu().numpy()

def look_at(view, target):
    d = (-target + view)
    d = d / np.linalg.norm(d)
    up = np.array([0, 0, 1])
    r = np.cross(up, d)
    r /= np.linalg.norm(r)
    u = np.cross(d, r)
    u /= np.linalg.norm(u)

    c2w = np.eye(4)
    c2w[:3, :3] = np.linalg.inv(np.stack([r, u, d], axis=0))
    c2w[:3, 3] = view

    c2w[:3, 1:3] *= -1  # Y and Z flip
    w2c = np.linalg.inv(c2w)

    R = w2c[:3, :3].T  # transpose for CUDA compatibility
    T = w2c[:3, 3]
    return R, T



def perturb_and_generate_poses(scene, gaussians, output_json_path, n_poses=10, radius_perturb=0.05):
    center = extract_object_center_from_gaussians(scene)
    train_cams = scene.getTrainCameras()
    poses = []

    # Step 0: Select a single base camera
    base_idx = np.random.choice(len(train_cams))
    base_cam = train_cams[base_idx]
    base_pos = base_cam.camera_center.cpu().numpy()

    direction = base_pos / np.linalg.norm(base_pos)
    radius = np.linalg.norm(base_pos)

    for i in range(n_poses):
        # Step 1: Perturb on tangent of sphere centered at origin
        tangent = np.random.randn(3)
        tangent -= tangent.dot(direction) * direction  # make tangent orthogonal
        tangent /= np.linalg.norm(tangent)
        tangent *= radius_perturb

        new_direction = direction + tangent
        new_direction /= np.linalg.norm(new_direction)

        # Constrain to top hemisphere
        if new_direction[1] < 0:
            new_direction[1] *= -1
            new_direction /= np.linalg.norm(new_direction)

        # Step 2: New camera position
        new_pos = radius * new_direction

        # Step 3: Look at object center
        R, T = look_at(new_pos, center)

        poses.append({
            "R": R.tolist(),
            "T": T.tolist(),
            "FoVx": float(base_cam.FoVx),
            "FoVy": float(base_cam.FoVy)
        })

    with open(output_json_path, 'w') as f:
        json.dump(poses, f, indent=4)

    return poses