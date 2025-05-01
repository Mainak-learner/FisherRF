import torch
from tqdm import tqdm
from typing import List
from scene import Scene
import numpy as np
import random
from gaussian_renderer import modified_render
from gaussian_renderer import modified_render
from scene.cameras import DummyCamera
from utils.graphics_utils import look_at_torch, uv2car_torch

def compute_fft_quality(image_tensor):
    grayscale = image_tensor.mean(dim=0)
    f = torch.fft.fft2(grayscale)
    fshift = torch.fft.fftshift(f)
    magnitude_spectrum = torch.abs(fshift)
    median_freq = torch.median(magnitude_spectrum)
    return median_freq.item()


def calculate_distance(cam1, cam2):
    return torch.norm(cam1.camera_center - cam2.camera_center)

class EntropySelector(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # Initialize any parameters needed for your entropy calculation
        self.num_candidates = getattr(args, 'num_candidates', 100)
        self.seed = getattr(args, 'seed', 0)
        self.debug = getattr(args, 'debug', False)
        self.distance_sigma = args.distance_sigma  # Controls the influence of distance
        self.views_added = 0
        self.use_scheduler = args.use_scheduler
    
    def optimize_entropy_guided_view(self, gaussians, scene, pipe, background, train_cameras, quality_scores, steps=200):
        means = gaussians.capture()[1]
        object_center = means.mean(dim=0).detach() 
        u = torch.tensor([0.5], device='cuda', requires_grad=True)
        v = torch.tensor([0.3], device='cuda', requires_grad=True)
        r = torch.tensor([4.0], device='cuda', requires_grad=True)

        optimizer = torch.optim.Adam([u, v], lr=1e-3)

        for _ in range(steps):
            optimizer.zero_grad()

            # Clamp to reachable region (upper hemisphere + user-defined v-range)
            u.data = torch.remainder(u.data, 1.0)
            v.data = torch.clamp(v.data, 0.01, 0.48)  # Upper hemisphere (0 ~ 0.5)

            # Compute camera center from spherical coords
            cam_pos = uv2car_torch(u, v) * r + object_center
            R = look_at_torch(cam_pos, object_center)
            test_view = DummyCamera(R, torch.zeros(3, device="cuda"), train_cameras[0])
            test_view.camera_center = cam_pos

            render_pkg = modified_render(test_view, gaussians, pipe, background)
            entropy = render_pkg["entropy"].mean()

            # Guidance weight: inverse FFT quality-weighted distance to training cams
            guidance = 0.0
            for cam, q in zip(train_cameras, quality_scores):
                dist = torch.norm(cam_pos - cam.camera_center)
                guidance += dist * (1 - q)  # (1 - q) favors bad-quality views

            score = -(guidance * entropy)
            score.backward()
            optimizer.step()

        return test_view

def nbvs(self, gaussians, scene, num_views, pipe, background, completion_rate=None, exit_func=None):
    train_cameras = scene.getTrainCameras()

    def fft_quality(image_tensor):
        grayscale = image_tensor.mean(dim=0)
        f = torch.fft.fft2(grayscale)
        fshift = torch.fft.fftshift(f)
        return torch.median(torch.abs(fshift)).item()

    quality_scores = [fft_quality(cam.original_image) for cam in train_cameras]
    q_norm = (np.array(quality_scores) - np.min(quality_scores)) / (np.max(quality_scores) - np.min(quality_scores))

    selected_views = []
    for _ in range(num_views):
        new_view = self.optimize_entropy_guided_view(gaussians, scene, pipe, background, train_cameras, q_norm)
        scene.train_cameras[1.0].append(new_view)
        selected_views.append(len(scene.getTrainCameras()) - 1)

    return selected_views
        
