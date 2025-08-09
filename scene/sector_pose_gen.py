import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def _filter_overlap_by_angle(test_centers, exclude_centers, angle_deg=1.0, eps=1e-8):
    """
    Remove test centers that are within `angle_deg` of any center in `exclude_centers`.
    Works even if radii differ (we normalize to unit vectors).
    """
    if exclude_centers is None or len(exclude_centers) == 0:
        return test_centers, torch.ones(len(test_centers), dtype=torch.bool, device=test_centers.device)

    # Normalize to unit vectors
    t = test_centers / (test_centers.norm(dim=1, keepdim=True) + eps)          # (Nt, 3)
    p = exclude_centers / (exclude_centers.norm(dim=1, keepdim=True) + eps)    # (Ne, 3)

    # Cosine threshold
    thr = torch.cos(torch.tensor(angle_deg * np.pi / 180.0, device=t.device))

    # Cosine similarity matrix (Nt, Ne)
    cos_sim = t @ p.T
    max_sim, _ = cos_sim.max(dim=1)

    keep_mask = max_sim <= thr
    return test_centers[keep_mask], keep_mask


def sample_uniform_sphere_views_disjoint(
    num_views=400,
    radius=1.0,
    object_center=torch.tensor([0.0, 0.0, 0.0]),
    exclude_centers=None,          # torch.Tensor of shape (Ne, 3), e.g. your proposal centers
    angle_deg=1.0,                 # min angular separation from excluded poses
    oversample_factor=2,           # generate more then filter to hit target count
):
    """
    Uniform Fibonacci sampling on a sphere, then drop any pose within `angle_deg`
    of any excluded pose (e.g., proposal poses). Returns exactly `num_views`
    if possible; otherwise returns as many as remain after filtering.
    """
    device = object_center.device
    N = int(num_views * oversample_factor)

    # Fibonacci sphere sampling
    indices = np.arange(0, N, dtype=np.float32) + 0.5
    phi = np.arccos(1 - 2 * indices / N)
    theta = np.pi * (1 + 5**0.5) * indices

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    centers_all = torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32, device=device)
    directions_all = F.normalize(object_center[None, :] - centers_all, dim=1)

    # Filter out overlaps with excluded centers
    if exclude_centers is not None and len(exclude_centers) > 0:
        centers_filt, keep_mask = _filter_overlap_by_angle(centers_all, exclude_centers, angle_deg=angle_deg)
        directions_filt = directions_all[keep_mask]
    else:
        centers_filt, directions_filt = centers_all, directions_all

    # If we still have more than needed, downselect uniformly
    if len(centers_filt) > num_views:
        # Pick a uniform subset (could also do farthest-point sampling if you prefer)
        idx = torch.linspace(0, len(centers_filt) - 1, steps=num_views, device=device).round().long()
        centers_filt = centers_filt[idx]
        directions_filt = directions_filt[idx]

    return centers_filt, directions_filt


def generate_circular_hemisphere_poses(center, num_circles=9, min_poses=30, radius=1.5):
    """
    Generate poses on a hemisphere.
    """
    assert min_poses % 6 == 0, "Min Poses must be divisible my 6"
    all_poses = []
    all_uvs = []
    pose_per_circle = []
    for i in range(num_circles):
        # Elevation angle (v): 0 = pole (top), pi/2 = equator (middle)
        elevation = np.pi / 2 * (i + 1) / (num_circles + 1)
        num_poses = int(min_poses * (i+1))
        pose_per_circle.append(num_poses)
        for j in range(num_poses):
            azimuth = 2 * np.pi * j / num_poses
            x = radius * np.sin(elevation) * np.cos(azimuth)
            y = radius * np.sin(elevation) * np.sin(azimuth)
            z = radius * np.cos(elevation)
            cam_center = torch.tensor([x, y, z], dtype=torch.float32, device=center.device) + center
            all_poses.append(cam_center)
            all_uvs.append((azimuth / (2*np.pi), elevation / np.pi))
    return torch.stack(all_poses), all_uvs, pose_per_circle

def divide_hemisphere_poses(poses_xyz, center, poses_per_circle, num_circles=9):
    """
    Divide camera poses into middle circle and 12 upper/lower sectors (6 each).
    """
    # assert poses_xyz.shape[0] == num_circles * num_poses_per_circle, "Expected 150 poses"

    # Directions from object center
    poses_np = poses_xyz.detach().cpu().numpy()  # <-- NEW
    directions = poses_np - center           # <-- FIXED

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms
    # azimuth = torch.atan2(directions[:, 1], directions[:, 0])  # [-pi, pi]
    # elevation = torch.asin(directions[:, 2])  # [-pi/2, pi/2]

    # Sector definitions
    middle_ring = num_circles//2  # third circle (index 2) is middle
    circle_indices = {}
    middle_circle_indices = []
    sector_map = {}
    pose_idx = 0

    for i in range(num_circles):
        circle_indices[f"elev_{i}"] = []
        for j in range(poses_per_circle[i]):
            if i == middle_ring:
                middle_circle_indices.append(pose_idx)
                pose_idx += 1
                continue

            sector_label = f"{'upper' if i < middle_ring else 'lower'}_{(j * 6) // poses_per_circle[i]}"
            if sector_label not in sector_map:
                sector_map[sector_label] = []
            # if len(sector_map[sector_label]) < 10:
            sector_map[sector_label].append(pose_idx)
            circle_indices[f"elev_{i}"].append(pose_idx)
            pose_idx += 1


    return circle_indices, middle_circle_indices, sector_map