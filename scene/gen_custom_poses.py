import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from scene import Scene
from gaussian_renderer import GaussianModel

def extract_object_center(scene):
    train_cameras = scene.getTrainCameras()
    centers = [cam.camera_center for cam in train_cameras]
    return np.mean(centers, axis=0)

def look_at(camera_pos, target):
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    tmp = np.array([0, 1, 0])
    right = np.cross(tmp, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack([right, up, forward], axis=1)

def perturb_and_generate_poses(scene, output_json_path, n_poses=10, radius_perturb=0.05):
    center = extract_object_center(scene)
    train_cams = scene.getTrainCameras()
    poses = []

    indices = np.random.choice(len(train_cams), size=n_poses, replace=True)
    for i in range(n_poses):
        cam = train_cams[indices[i]]
        base_pos = cam.camera_center
        direction = base_pos - center
        radius = np.linalg.norm(direction)
        direction /= radius  # unit vector

        # Tangential perturbation
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

        new_pos = center + radius * new_direction
        R_mat = look_at(new_pos, center)

        poses.append({
            "R": R_mat.tolist(),
            "T": new_pos.tolist(),
            "FoVx": cam.FoVx,
            "FoVy": cam.FoVy
        })

    with open(output_json_path, 'w') as f:
        json.dump(poses, f, indent=4)