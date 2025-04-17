import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R

def extract_object_center(train_json_path):
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    centers = []
    for frame in train_data['frames']:
        transform = np.array(frame['transform_matrix'])  # (4x4)
        centers.append(transform[:3, 3])  # translation vector
    return np.mean(centers, axis=0)

def look_at(camera_pos, target):
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    tmp = np.array([0, 1, 0])
    right = np.cross(tmp, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R_mat = np.stack([right, up, forward], axis=1)
    return R_mat

def generate_spherical_poses(center, radius=4.0, n_poses=10):
    poses = []
    for i in range(n_poses):
        theta = 2 * np.pi * i / n_poses
        phi = np.pi / 6 + np.random.uniform(-0.1, 0.1)  # elevation
        cam_x = radius * np.sin(phi) * np.cos(theta)
        cam_y = radius * np.cos(phi)
        cam_z = radius * np.sin(phi) * np.sin(theta)
        cam_pos = center + np.array([cam_x, cam_y, cam_z])
        R_mat = look_at(cam_pos, center)
        T = cam_pos
        poses.append({"R": R_mat.tolist(), "T": T.tolist(),
                      "FoVx": 0.6911, "FoVy": 0.6911})
    return poses

def save_generated_poses(output_path, poses):
    with open(output_path, 'w') as f:
        json.dump(poses, f, indent=4)

def generate_custom_poses_from_train(train_json_path, output_json_path, radius=4.0, n_poses=10):
    center = extract_object_center(train_json_path)
    poses = generate_spherical_poses(center, radius=radius, n_poses=n_poses)
    save_generated_poses(output_json_path, poses)
