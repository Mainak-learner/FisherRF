import numpy as np
import json
from scipy.spatial.transform import Rotation as R

def extract_object_center(train_json_path):
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    centers = [np.array(frame['transform_matrix'])[:3, 3] for frame in train_data['frames']]
    return np.mean(centers, axis=0)

def look_at(camera_pos, target):
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    tmp = np.array([0, 1, 0])
    right = np.cross(tmp, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack([right, up, forward], axis=1)

def perturb_and_generate_poses(train_json_path, output_json_path, n_poses=10, radius_perturb=0.2, angle_perturb_deg=5.0):
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

    center = extract_object_center(train_json_path)
    poses = []

    indices = np.random.choice(len(train_data['frames']), size=n_poses, replace=True)
    for i in range(n_poses):
        base_frame = train_data['frames'][indices[i]]
        transform = np.array(base_frame['transform_matrix'])  # 4x4

        # Camera position
        base_pos = transform[:3, 3]

        # Add small random radius perturbation
        direction = base_pos - center
        direction /= np.linalg.norm(direction)
        perturbed_distance = np.linalg.norm(base_pos - center) + np.random.uniform(-radius_perturb, radius_perturb)
        new_pos = center + direction * perturbed_distance

        # Apply small random rotation (in degrees)
        angle = np.random.uniform(-angle_perturb_deg, angle_perturb_deg)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        rot = R.from_rotvec(np.deg2rad(angle) * axis)

        direction_rotated = rot.apply(direction)
        new_pos = center + direction_rotated * perturbed_distance

        # Look at center
        R_mat = look_at(new_pos, center)

        poses.append({
            "R": R_mat.tolist(),
            "T": new_pos.tolist(),
            "FoVx": base_frame.get("fov_x", 0.6911),
            "FoVy": base_frame.get("fov_y", 0.6911)
        })

    with open(output_json_path, 'w') as f:
        json.dump(poses, f, indent=4)
