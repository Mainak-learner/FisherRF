import torch
import numpy as np
from scipy.spatial.distance import cdist
from torch.nn import Module

class GPFisherNBVSelector(Module):
    def __init__(self, device="cuda", lengthscale=0.15, sigma_f=1.0, noise=1e-6):
        super().__init__()
        self.device = device
        self.l = lengthscale
        self.sigma_f = sigma_f
        self.noise = noise

    def rbf_kernel(self, X1, X2):
        dists = cdist(X1.cpu().numpy(), X2.cpu().numpy(), 'sqeuclidean')
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l**2 * dists)

    def compute_uncertainty_trace(self, proposal_centers, object_center, gaussians, pipe, background, reference_camera, fim_mode="trace"):
        from utils.camera_utils import look_at
        from scene.cameras import DummyCamera
        from gaussian_renderer import render

        uncertainties = []
        for cam_center in proposal_centers:
            R, T = look_at(cam_center.detach(), object_center.detach())
            dummy_cam = DummyCamera(R, T, reference_camera)
            render_pkg = render(dummy_cam, gaussians, pipe, background)
            image = render_pkg["render"]
            image.backward(gradient=torch.ones_like(image))
            fim = torch.cat([p.grad.detach().reshape(-1) for p in gaussians.capture()[1:7]])
            uncertainty = fim.norm() if fim_mode == "norm" else fim.sum()
            uncertainties.append(uncertainty.item())
            gaussians.optimizer.zero_grad(set_to_none=True)
        return torch.tensor(uncertainties, device=self.device)

    def optimize(self, proposal_uvs, proposal_centers, uncertainties, all_uvs, all_centers):
        # GP posterior mean and variance
        X_train = torch.tensor(proposal_uvs, dtype=torch.float32, device=self.device)
        y_train = uncertainties
        X_test = torch.tensor(all_uvs, dtype=torch.float32, device=self.device)

        K = self.rbf_kernel(X_train, X_train)
        K += self.noise * np.eye(K.shape[0])
        K_s = self.rbf_kernel(X_train, X_test)
        K_ss = self.rbf_kernel(X_test, X_test)
        K_inv = np.linalg.inv(K)

        mu = K_s.T @ K_inv @ y_train.cpu().numpy()
        # Find pose with max predicted uncertainty
        best_idx = np.argmax(mu)
        return all_centers[best_idx], all_uvs[best_idx]
