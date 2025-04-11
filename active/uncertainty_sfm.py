# utils/uncertainty_sfm.py

import cv2
import numpy as np
import torch
from utils.camera_utils import rand_rotation_matrix
from utils.graphics_utils import getWorld2View
from scene.cameras import Camera
from gaussian_renderer import render
from typing import List, Tuple

def generate_perturbed_poses(original_camera: Camera, num_perturbations: int = 5, deflection: float = 0.1, translation_magnitude: float = 0.1) -> List[Camera]:
    perturbed_cameras = []
    R = original_camera.R.detach().cpu().numpy() if torch.is_tensor(original_camera.R) else np.array(original_camera.R)
    T = original_camera.T.detach().cpu().numpy() if torch.is_tensor(original_camera.T) else np.array(original_camera.T)
    height, width = original_camera.image_height, original_camera.image_width
    trans = original_camera.trans.detach().cpu().numpy() if torch.is_tensor(original_camera.trans) else np.array(original_camera.trans)
    scale = original_camera.scale

    for i in range(num_perturbations):
        R_perturb = rand_rotation_matrix(deflection=deflection)
        R_new = R_perturb @ R
        T_perturb = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
        T_new = T + T_perturb
        perturbed_cam = Camera(
            colmap_id=original_camera.colmap_id,
            R=R_new,
            T=T_new,
            FoVx=original_camera.FoVx,
            FoVy=original_camera.FoVy,
            image=None,
            gt_alpha_mask=None,
            image_name=f"perturbed_{i}",
            uid=original_camera.uid,
            trans=trans,
            scale=scale,
            data_device="cuda",
            height=height,
            width=width
        )
        perturbed_cameras.append(perturbed_cam)
    return perturbed_cameras

def render_perturbed_images(original_camera: Camera, perturbed_cameras: List[Camera], gaussians, pipe, background) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if original_camera.original_image is not None:
        original_render = render(original_camera, gaussians, pipe, background)["render"]
    else:
        original_render = torch.zeros((3, original_camera.image_height, original_camera.image_width), device="cuda")

    perturbed_renders = [render(cam, gaussians, pipe, background)["render"] for cam in perturbed_cameras]
    return original_render, perturbed_renders

def extract_features(img: torch.Tensor) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    # Ensure img is a PyTorch tensor and detach it if it requires gradients
    if not torch.is_tensor(img):
        img = torch.from_numpy(np.array(img)).float()
    if img.requires_grad:
        img = img.detach()  # Detach from computational graph
    if img.device != torch.device("cpu"):
        img = img.cpu()  # Move to CPU if not already there
    # Convert from (C, H, W) to (H, W, C) and scale to 0-255
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def estimate_relative_poses(original_image: torch.Tensor, perturbed_images: List[torch.Tensor]) -> List[Tuple[np.ndarray, np.ndarray]]:
    original_kp, original_desc = extract_features(original_image)
    relative_poses = []

    for perturbed_img in perturbed_images:
        perturbed_kp, perturbed_desc = extract_features(perturbed_img)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(original_desc, perturbed_desc, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) < 8:
            relative_poses.append((np.eye(3), np.zeros(3)))
            continue

        src_pts = np.float32([original_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([perturbed_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or mask is None:
            relative_poses.append((np.eye(3), np.zeros(3)))
            continue
        _, R_est, T_est, _ = cv2.recoverPose(E, src_pts, dst_pts, focal=1.0, pp=(0., 0.))
        relative_poses.append((R_est, T_est))

    return relative_poses

def compute_uncertainty(actual_poses: List[Tuple[np.ndarray, np.ndarray]], estimated_poses: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    total_error = 0.0
    num_valid = 0
    for actual_R, actual_T, est_R, est_T in zip([p[0] for p in actual_poses], [p[1] for p in actual_poses], 
                                                [p[0] for p in estimated_poses], [p[1] for p in estimated_poses]):
        if np.array_equal(est_R, np.eye(3)) and np.array_equal(est_T, np.zeros(3)):
            continue
        R_diff = np.dot(actual_R.T, est_R)
        angle_error = np.arccos((np.trace(R_diff) - 1) / 2) * 180 / np.pi
        T_error = np.linalg.norm(actual_T - est_T)
        total_error += angle_error + T_error
        num_valid += 1
    return total_error / max(1, num_valid)

def evaluate_pose_uncertainty(original_camera: Camera, gaussians, pipe, background, num_perturbations: int = 5, 
                             deflection: float = 0.1, translation_magnitude: float = 0.1) -> float:
    perturbed_cams = generate_perturbed_poses(original_camera, num_perturbations, deflection, translation_magnitude)
    original_render, perturbed_renders = render_perturbed_images(original_camera, perturbed_cams, gaussians, pipe, background)
    actual_relative_poses = [(cam.R.detach().cpu().numpy() if torch.is_tensor(cam.R) else np.array(cam.R), 
                              cam.T.detach().cpu().numpy() if torch.is_tensor(cam.T) else np.array(cam.T)) 
                             for cam in perturbed_cams]
    estimated_relative_poses = estimate_relative_poses(original_render, perturbed_renders)
    uncertainty = compute_uncertainty(actual_relative_poses, estimated_relative_poses)
    return uncertainty