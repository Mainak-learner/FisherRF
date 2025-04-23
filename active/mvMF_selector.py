import torch
import numpy as np
from torch.distributions import Categorical
from scipy.special import iv
from utils.image_utils import psnr
from gaussian_renderer import render

def vmf_normalization_const(kappa, dim=3):
    return kappa**(dim/2 - 1) / ((2 * np.pi)**(dim / 2) * iv(dim/2 - 1, kappa))

def vmf_density(x, mu, kappa):
    # Assumes x and mu are unit vectors
    return vmf_normalization_const(kappa) * np.exp(kappa * np.dot(mu, x))

def get_camera_centers(cameras):
    centers = []
    for cam in cameras:
        centers.append(cam.camera_center.detach().cpu().numpy())
    return centers

class MvMFSelector:
    def __init__(self, args):
        self.kappa = args.kappa if hasattr(args, "kappa") else 10.0
        self.temperature = args.sampling_temperature if hasattr(args, "sampling_temperature") else 0.07

    def compute_view_errors_psnr(self, gaussians, scene, pipe, background):
        errors = []
        train_cameras = scene.getTrainCameras()
        for cam in train_cameras:
            rendered = torch.clamp(
                render(cam, gaussians, pipe, background)["render"], 0.0, 1.0
            )
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            value = psnr(rendered, gt).mean().item()
            errors.append(1.0 / value)  # Inverse PSNR as error

        return np.array(errors)

    def nbvs(self, gaussians, scene, num_views, pipe, background, exit_func=None):
        V = scene.getTrainCameras()
        candidate_views = list(scene.get_candidate_set())
        candidate_cams = scene.getCandidateCameras()
        
        camera_centers = get_camera_centers(V)
        candidate_centers = get_camera_centers(candidate_cams)
        
        # Assume scene.errors has the per-view error values
        m = self.compute_view_errors_psnr(gaussians, scene, pipe, background)
        print(m)
        m_hat = (np.max(m) - m) / (np.max(m) - m + 1e-6)
        alpha = torch.softmax(torch.tensor(m_hat) / self.temperature, dim=0).numpy()

        # Sample view index from categorical dist
        selected_ids = []
        for _ in range(num_views):
            component_idx = np.random.choice(len(V), p=alpha)
            mu = camera_centers[component_idx]
            mu = mu / np.linalg.norm(mu)
            x = self.sample_vmf(mu, self.kappa)
            
            # Choose closest candidate view to x
            dists = [np.linalg.norm(x - v / np.linalg.norm(v)) for v in candidate_centers]
            closest_idx = np.argmin(dists)
            selected_ids.append(closest_idx)
        return [candidate_views[i] for i in selected_ids]

    def sample_vmf(self, mu, kappa):
        dim = mu.shape[0]
        w = self._sample_weight(kappa, dim)
        v = self._sample_orthonormal_to(mu)
        return np.sqrt(1 - w**2) * v + w * mu

    def _sample_weight(self, kappa, dim):
        b = (-2 * kappa + np.sqrt(4 * kappa**2 + (dim - 1)**2)) / (dim - 1)
        x = (1 - b) / (1 + b)
        c = kappa * x + (dim - 1) * np.log(1 - x**2)
        while True:
            z = np.random.beta((dim - 1) / 2, (dim - 1) / 2)
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            u = np.random.uniform()
            if kappa * w + (dim - 1) * np.log(1 - x * w) - c >= np.log(u):
                return w

    def _sample_orthonormal_to(self, mu):
        v = np.random.randn(*mu.shape)
        proj = v - mu * np.dot(mu, v)
        return proj / np.linalg.norm(proj)
