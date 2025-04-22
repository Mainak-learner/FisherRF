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

def perturb_and_generate_poses(train_json_path, output_json_path, n_poses=10, radius_perturb=0.05):
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

    center = extract_object_center(train_json_path)
    poses = []

    indices = np.random.choice(len(train_data['frames']), size=n_poses, replace=True)
    for i in range(n_poses):
        base_frame = train_data['frames'][indices[i]]
        transform = np.array(base_frame['transform_matrix'])  # 4x4

        base_pos = transform[:3, 3]
        direction = base_pos - center
        radius = np.linalg.norm(direction)
        direction /= radius  # unit direction

        # Tangent perturbation in top hemisphere
        tangent = np.random.randn(3)
        tangent -= tangent.dot(direction) * direction  # project onto tangent plane
        tangent /= np.linalg.norm(tangent)
        tangent *= radius_perturb

        new_direction = direction + tangent
        new_direction /= np.linalg.norm(new_direction)

        # Ensure top hemisphere: positive Y in Blender coordinates
        if new_direction[1] < 0:
            new_direction[1] *= -1
            new_direction /= np.linalg.norm(new_direction)

        new_pos = center + radius * new_direction

        # Compute look-at rotation matrix
        R_mat = look_at(new_pos, center)

        poses.append({
            "R": R_mat.tolist(),
            "T": new_pos.tolist(),
            "FoVx": base_frame.get("fov_x", 0.6911),
            "FoVy": base_frame.get("fov_y", 0.6911)
        })

    with open(output_json_path, 'w') as f:
        json.dump(poses, f, indent=4)
