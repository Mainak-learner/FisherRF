import torch
import numpy as np
from scipy.spatial.distance import cdist
from typing import List
from torch.nn import Module
from tqdm import tqdm
from scene import Scene
from utils.camera_utils import look_at, look_at_torch
from utils.graphics_utils import uv2car_torch
from gaussian_renderer import render, network_gui, modified_render

class GPFisherNBVSelector(Module):
    def __init__(self, args, device="cuda", lengthscale=0.15, sigma_f=1.0, noise=1e-6):
        super().__init__()
        self.device = device
        self.l = lengthscale
        self.sigma_f = sigma_f
        self.noise = noise

        self.seed = args.seed
        self.reg_lambda = args.reg_lambda
        self.I_test: bool = args.I_test
        self.I_acq_reg: bool = args.I_acq_reg

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx: List[str] = [name2idx[k] for k in args.filter_out_grad]

    def rbf_kernel(self, X1, X2):
        dists = cdist(X1.detach().cpu().numpy(), X2.detach().cpu().numpy(), 'sqeuclidean')
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l**2 * dists)

    def compute_fisher_uncertainty(self, gaussians, selected_cameras, candidate_cameras, pipe, background):
        """
        Computes acquisition scores for each camera pose as the uncertainty estimate.
        Returns:
            torch.Tensor of shape (N,) where N is the number of candidate_cameras
        """
        params = gaussians.capture()[1:7]
        params = [p for i, p in enumerate(params) if i not in self.filter_out_idx]
        device = params[0].device

        H_train = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype)

        # Use training or test cameras to build H_train
        viewpoint_cams = selected_cameras
        for cam in tqdm(viewpoint_cams, desc="Calculating diagonal Hessian on training views"):
            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

            H_train += cur_H

            gaussians.optimizer.zero_grad(set_to_none = True) 

        H_train = H_train.to(device)

        I_train = torch.reciprocal(H_train + self.reg_lambda)
        acq_scores = torch.zeros(len(candidate_cameras))

        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating diagonal Hessian on proposal views")):

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

            I_acq = cur_H

            if self.I_acq_reg:
                I_acq += self.reg_lambda

            gaussians.optimizer.zero_grad(set_to_none = True) 
            acq_scores[idx] += torch.sum(I_acq * I_train).item()

        return torch.tensor(acq_scores, device=params[0].device)

    def optimize_gp_posterior(self, proposal_uvs, proposal_centers, uncertainties, init_uv, uv_bounds, radius, steps=100, lr=1e-2):
        """
        Args:
            proposal_uvs: (N, 2) list of (u, v) for training proposals
            proposal_centers: (N, 3) world cam centers corresponding to proposal_uvs
            uncertainties: (N,) tensor - uncertainty values at proposal poses
            init_uv: (u, v) tuple - initial uv to optimize from
            uv_bounds: ((u_min, u_max), (v_min, v_max))
            radius: float - fixed radius
            steps: int - number of gradient ascent steps
            lr: float - learning rate
        Returns:
            optimized_cam_center: (3,) tensor
            optimized_uv: (u, v) tuple
        """
        device = self.device
        u_min, u_max = uv_bounds[0]
        v_min, v_max = uv_bounds[1]

        u = torch.tensor([init_uv[0]], device=device, dtype=torch.float32, requires_grad=True)
        v = torch.tensor([init_uv[1]], device=device, dtype=torch.float32, requires_grad=True)

        X_train = torch.tensor(proposal_centers, dtype=torch.float32, device=device)
        y_train = uncertainties.detach()
        K = self.rbf_kernel(X_train, X_train)
        K += self.noise * np.eye(K.shape[0])
        K_inv = torch.tensor(np.linalg.inv(K), device=device, dtype=torch.float32)
        y_train = y_train.unsqueeze(1)

        optimizer = torch.optim.Adam([u, v], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            # Convert u,v to camera center
            cam_center = uv2car_torch(u, v).to(device) * radius  # (1, 3)
            K_s = self.rbf_kernel(X_train, cam_center)  # shape (N, 1)
            mu = K_s.T @ K_inv @ y_train  # scalar
            loss = -mu  # maximize mu
            loss.backward()
            optimizer.step()

            # Clamp to valid sector bounds
            u.data.clamp_(u_min, u_max)
            v.data.clamp_(v_min, v_max)

        final_uv = (u.item(), v.item())
        final_center = uv2car_torch(u, v).squeeze(0).detach().cpu()
        return final_center, final_uv

