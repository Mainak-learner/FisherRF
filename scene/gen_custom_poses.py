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



def perturb_and_generate_poses(scene, gaussians, output_json_path, angle_perturb=np.deg2rad(5)):
    center = extract_object_center_from_gaussians(scene)
    train_cams = scene.getTrainCameras()
    poses = []

    # Select a single base camera
    base_idx = np.random.choice(len(train_cams))
    base_cam = train_cams[base_idx]
    base_pos = base_cam.camera_center.cpu().numpy()

    # Normalize base direction and compute spherical angles
    direction = base_pos / np.linalg.norm(base_pos)
    radius = np.linalg.norm(base_pos)

    x, y, z = direction
    theta = np.arccos(z)               # polar angle (from +z)
    phi = np.arctan2(y, x)             # azimuth angle

    delta = angle_perturb

    # Define angular perturbations
    perturbations = {
        "theta_up":    (theta - delta, phi),
        "theta_down":  (theta + delta, phi),
        "phi_left":    (theta, phi - delta),
        "phi_right":   (theta, phi + delta),
    }

    for name, (theta_p, phi_p) in perturbations.items():
        # Clamp theta to avoid poles
        theta_p = np.clip(theta_p, 1e-4, np.pi - 1e-4)

        # Convert spherical → cartesian
        dx = np.sin(theta_p) * np.cos(phi_p)
        dy = np.sin(theta_p) * np.sin(phi_p)
        dz = np.cos(theta_p)
        new_dir = np.array([dx, dy, dz])
        new_pos = radius * new_dir

        R, T = look_at(new_pos, center)

        poses.append({
            "name": name,
            "R": R.tolist(),
            "T": T.tolist(),
            "FoVx": float(base_cam.FoVx),
            "FoVy": float(base_cam.FoVy)
        })

    with open(output_json_path, 'w') as f:
        json.dump(poses, f, indent=4)

    return poses