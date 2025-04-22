import numpy as np
import json
from scipy.spatial.transform import Rotation as R

def extract_object_center_from_gaussians(scene):
    means = list(scene.gaussians.capture()[1:7])
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

    indices = np.random.choice(len(train_cams), size=n_poses, replace=True)
    for i in range(n_poses):
        cam = train_cams[indices[i]]
        base_pos = cam.camera_center.cpu().numpy()
        
        # Step 1: Use direction from ORIGIN
        direction = base_pos / np.linalg.norm(base_pos)
        radius = np.linalg.norm(base_pos)

        # Step 2: Perturb on tangent of sphere centered at origin
        tangent = np.random.randn(3)
        tangent -= tangent.dot(direction) * direction
        tangent /= np.linalg.norm(tangent)
        tangent *= radius_perturb

        new_direction = direction + tangent
        new_direction /= np.linalg.norm(new_direction)

        # Constrain to top hemisphere
        if new_direction[1] < 0:
            new_direction[1] *= -1
            new_direction /= np.linalg.norm(new_direction)

        # Step 3: New camera position on sphere centered at origin
        new_pos = radius * new_direction

        # Step 4: Look at actual object center (possibly ≠ origin)
        R, T = look_at(new_pos, center)

        poses.append({
            "R": R.tolist(),
            "T": T.tolist(),
            "FoVx": float(cam.FoVx),
            "FoVy": float(cam.FoVy)
        })

    with open(output_json_path, 'w') as f:
        json.dump(poses, f, indent=4)
