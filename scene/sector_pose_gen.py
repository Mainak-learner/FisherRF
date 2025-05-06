import numpy as np
import torch

def generate_circular_hemisphere_poses(center, num_circles=5, num_poses_per_circle=30, radius=1.5):
    """
    Generate 150 poses on a hemisphere (5 circles × 30 poses each)
    """
    all_poses = []
    all_uvs = []
    for i in range(num_circles):
        # Elevation angle (v): 0 = pole (top), pi/2 = equator (middle)
        elevation = np.pi / 2 * (i + 1) / (num_circles + 1)
        for j in range(num_poses_per_circle):
            azimuth = 2 * np.pi * j / num_poses_per_circle
            x = radius * np.sin(elevation) * np.cos(azimuth)
            y = radius * np.sin(elevation) * np.sin(azimuth)
            z = radius * np.cos(elevation)
            cam_center = torch.tensor([x, y, z], dtype=torch.float32) + center
            all_poses.append(cam_center)
            all_uvs.append((azimuth / (2*np.pi), elevation / np.pi))
    return torch.stack(all_poses), all_uvs

def divide_hemisphere_poses(poses_xyz, center, num_circles=5, num_poses_per_circle=30):
    """
    Divide 150 camera poses into middle circle and 12 upper/lower sectors (6 each).
    Assumes poses are uniformly sampled across 5 elevation rings with 30 azimuthal angles each.
    """
    assert poses_xyz.shape[0] == num_circles * num_poses_per_circle, "Expected 150 poses"

    # Directions from object center
    directions = poses_xyz - center
    directions = directions / torch.norm(directions, dim=1, keepdim=True)
    azimuth = torch.atan2(directions[:, 1], directions[:, 0])  # [-pi, pi]
    elevation = torch.asin(directions[:, 2])  # [-pi/2, pi/2]

    # Sector definitions
    middle_ring = 2  # third circle (index 2) is middle
    circle_indices = {}
    middle_circle_indices = []
    sector_map = {}

    for i in range(num_circles):
        for j in range(num_poses_per_circle):
            idx = i * num_poses_per_circle + j
            if i == middle_ring:
                middle_circle_indices.append(idx)
                continue

            sector_label = f"{'upper' if i < middle_ring else 'lower'}_{(j * 6) // num_poses_per_circle}"
            if sector_label not in sector_map:
                sector_map[sector_label] = []
            if len(sector_map[sector_label]) < 10:
                sector_map[sector_label].append(idx)

    # Create elevation groupings for completeness
    for i in range(num_circles):
        circle_indices[f"elev_{i}"] = [i * num_poses_per_circle + j for j in range(num_poses_per_circle)]

    return circle_indices, middle_circle_indices, sector_map