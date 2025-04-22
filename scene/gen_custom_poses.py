import numpy as np
import json
from scipy.spatial.transform import Rotation as R

def extract_object_center(scene):
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    centers = [cam.camera_center.cpu().numpy() for cam in train_cameras] + [cam.camera_center.cpu().numpy() for cam in test_cameras]
    return np.mean(centers, axis=0)

def look_at(camera_pos, target, up=np.array([0, 1, 0])):
    forward = (target - camera_pos)
    forward /= np.linalg.norm(forward)

    if np.abs(np.dot(forward, up)) > 0.99:
        up = np.array([0, 0, 1])  # fallback to avoid degenerate cross-product

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    new_up = np.cross(forward, right)

    return np.stack([right, new_up, forward], axis=1)


def perturb_and_generate_poses(scene, output_json_path, n_poses=10, radius_perturb=0.05):
    center = extract_object_center(scene)
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
        R_mat = look_at(new_pos, center)

        poses.append({
            "R": R_mat.tolist(),
            "T": new_pos.tolist(),
            "FoVx": float(cam.FoVx),
            "FoVy": float(cam.FoVy)
        })

    with open(output_json_path, 'w') as f:
        json.dump(poses, f, indent=4)
