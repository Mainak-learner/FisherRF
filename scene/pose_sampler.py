# pose_sampler.py
import numpy as np

def divide_hemisphere_poses(poses, center, circle_elevations=[-0.6, -0.3, 0.0, 0.3, 0.6], epsilon=0.05):
    directions = poses[:, :3, 3] - center
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_directions = directions / norms

    azimuth = np.arctan2(unit_directions[:, 1], unit_directions[:, 0])  # [-pi, pi]
    elevation = np.arcsin(unit_directions[:, 2])  # [-pi/2, pi/2]

    # Step 1: Group poses into 5 horizontal elevation circles
    circle_indices = {}
    for elev in circle_elevations:
        mask = np.abs(elevation - elev) < epsilon
        key = f'elev_{elev:.2f}'
        circle_indices[key] = np.where(mask)[0]

    # Step 2: Identify the middle circle (elevation closest to 0)
    middle_key = min(circle_indices.keys(), key=lambda k: abs(float(k.split('_')[1])))
    middle_circle_indices = circle_indices[middle_key]

    # Step 3: Build upper/lower hemisphere masks
    upper_mask = elevation > float(middle_key.split('_')[1]) + epsilon
    lower_mask = elevation < float(middle_key.split('_')[1]) - epsilon

    upper_az = azimuth[upper_mask]
    lower_az = azimuth[lower_mask]

    def assign_sector(az_array, count=6):
        return np.floor((az_array + np.pi) / (2 * np.pi) * count).astype(int)

    upper_sectors = assign_sector(upper_az)
    lower_sectors = assign_sector(lower_az)

    upper_indices = np.where(upper_mask)[0]
    lower_indices = np.where(lower_mask)[0]

    sector_map = {}
    for i in range(6):
        sector_map[f'upper_{i}'] = upper_indices[upper_sectors == i]
        sector_map[f'lower_{i}'] = lower_indices[lower_sectors == i]

    return circle_indices, middle_circle_indices, sector_map