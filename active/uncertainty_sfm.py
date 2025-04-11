import cv2
import numpy as np
import torch
from utils.camera_utils import rand_rotation_matrix
from utils.graphics_utils import getWorld2View
from scene.cameras import Camera
from gaussian_renderer import render
from typing import List, Tuple

def generate_perturbed_poses(original_camera: Camera, num_perturbations: int = 5, deflection: float = 0.1, translation_magnitude: float = 0.1) -> List[Camera]:
    """
    Generate perturbed camera poses around the original camera pose.
    Args:
        original_camera: Camera object from which to perturb (assumed to have dataset dimensions).
        num_perturbations: Number of perturbed poses to generate.
        deflection: Magnitude of rotation perturbation (0 to 1).
        translation_magnitude: Magnitude of translation perturbation in world units.
    Returns:
        List of perturbed Camera objects with dataset-consistent dimensions.
    """
    perturbed_cameras = []
    # Extract R, T, trans, and scale as NumPy arrays for perturbation
    R = original_camera.R.detach().cpu().numpy()  # Detach tensor and convert to NumPy
    T = original_camera.T.detach().cpu().numpy()
    height, width = original_camera.image_height, original_camera.image_width  # Dataset dimensions
    trans = original_camera.trans  # Already a tensor in Camera
    scale = original_camera.scale  # Scalar, no conversion needed

    for i in range(num_perturbations):
        # Generate random rotation
        R_perturb = rand_rotation_matrix(deflection=deflection)
        R_new = R_perturb @ R

        # Generate random translation
        T_perturb = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
        T_new = T + T_perturb

        # Create new camera with perturbed pose, no image, but with dataset dimensions and original trans/scale
        perturbed_cam = Camera(
            colmap_id=original_camera.colmap_id,
            R=R_new,
            T=T_new,
            FoVx=original_camera.FoVx,
            FoVy=original_camera.FoVy,
            image=None,  # No ground truth image for perturbed poses
            gt_alpha_mask=None,
            image_name=f"perturbed_{i}",
            uid=original_camera.uid,
            trans=trans,  # Pass original trans as NumPy array
            scale=scale,  # Pass original scale
            data_device="cuda",
            height=height,  # Pass dataset height
            width=width     # Pass dataset width
        )
        perturbed_cameras.append(perturbed_cam)

    return perturbed_cameras

def render_perturbed_images(original_camera: Camera, perturbed_cameras: List[Camera], gaussians, pipe, background) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Render images from original and perturbed camera poses.
    If camera.image is None, it will be rendered later.
    """
    # Render original image if it exists, otherwise use a dummy tensor with correct dimensions
    if original_camera.image is not None:
        original_render = render(original_camera, gaussians, pipe, background)["render"]
    else:
        original_render = torch.zeros((3, original_camera.image_height, original_camera.image_width), device="cuda")  # Dummy render

    # Render perturbed images
    perturbed_renders = []
    for cam in perturbed_cameras:
        render_output = render(cam, gaussians, pipe, background)
        perturbed_renders.append(render_output["render"])

    return original_render, perturbed_renders

def extract_features(img: torch.Tensor) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extract SIFT features from a rendered image.
    """
    # Convert tensor to numpy array and then to grayscale
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def estimate_relative_poses(original_image: torch.Tensor, perturbed_images: List[torch.Tensor]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate relative poses using SfM (feature matching and pose estimation).
    Returns estimated rotation and translation matrices.
    """
    original_kp, original_desc = extract_features(original_image)
    relative_poses = []

    for perturbed_img in perturbed_images:
        perturbed_kp, perturbed_desc = extract_features(perturbed_img)

        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(original_desc, perturbed_desc, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:  # Need at least 8 points for essential matrix
            relative_poses.append((np.eye(3), np.zeros(3)))  # Default to identity if no good matches
            continue

        # Extract matched points
        src_pts = np.float32([original_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([perturbed_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Estimate essential matrix and recover pose
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or mask is None:
            relative_poses.append((np.eye(3), np.zeros(3)))  # Default to identity if estimation fails
            continue

        _, R_est, T_est, _ = cv2.recoverPose(E, src_pts, dst_pts, focal=1.0, pp=(0., 0.))

        relative_poses.append((R_est, T_est))

    return relative_poses

def compute_uncertainty(actual_poses: List[Tuple[np.ndarray, np.ndarray]], estimated_poses: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Compute uncertainty as the difference between actual and estimated relative poses.
    """
    total_error = 0.0
    num_valid = 0

    for actual_R, actual_T, est_R, est_T in zip([p[0] for p in actual_poses], [p[1] for p in actual_poses],
                                                [p[0] for p in estimated_poses], [p[1] for p in estimated_poses]):
        # Skip if either pose is identity (indicating failure)
        if np.array_equal(est_R, np.eye(3)) and np.array_equal(est_T, np.zeros(3)):
            continue

        # Rotation error (angle difference)
        R_diff = np.dot(actual_R.T, est_R)
        angle_error = np.arccos((np.trace(R_diff) - 1) / 2) * 180 / np.pi

        # Translation error (Euclidean distance)
        T_error = np.linalg.norm(actual_T - est_T)

        total_error += angle_error + T_error
        num_valid += 1

    return total_error / max(1, num_valid)  # Avoid division by zero

def evaluate_pose_uncertainty(original_camera: Camera, gaussians, pipe, background, num_perturbations: int = 5, 
                             deflection: float = 0.1, translation_magnitude: float = 0.1) -> float:
    """
    Evaluate uncertainty for a single camera pose using SfM.
    """
    # Generate perturbed poses
    perturbed_cams = generate_perturbed_poses(original_camera, num_perturbations, deflection, translation_magnitude)

    # Render images
    original_render, perturbed_renders = render_perturbed_images(original_camera, perturbed_cams, gaussians, pipe, background)

    # Get actual relative poses (from perturbed cameras)
    actual_relative_poses = [(cam.R, cam.T) for cam in perturbed_cams]

    # Estimate relative poses using SfM
    estimated_relative_poses = estimate_relative_poses(original_render, perturbed_renders)

    # Compute uncertainty
    uncertainty = compute_uncertainty(actual_relative_poses, estimated_relative_poses)

    return uncertainty