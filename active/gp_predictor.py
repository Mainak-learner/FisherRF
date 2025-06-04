import torch
import numpy as np
from scipy.spatial.distance import cdist
from typing import List
from torch.nn import Module
import torch.nn as nn
from tqdm import tqdm
from scene import Scene
from utils.camera_utils import look_at, look_at_torch
from utils.graphics_utils import uv2car_torch
from gaussian_renderer import render, network_gui, modified_render
import pyro
import pyro.contrib.gp as gp
from pyro.infer import Trace_ELBO
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

class VDGPFisherNBVSelector(Module):
    def __init__(self, args, input_dim, hidden_dim=32, num_inducing=128, device="cuda"):
        super().__init__()
        self.device = device
        self.input_dim = 5
        self.hidden_dim = hidden_dim

        self.seed = args.seed
        self.reg_lambda = args.reg_lambda
        self.I_test: bool = args.I_test
        self.I_acq_reg: bool = args.I_acq_reg

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx: List[str] = [name2idx[k] for k in args.filter_out_grad]

        self.latent_dim1 = 8  # or 16

        # 1st GP: 3D → 1D latent
        self.gp1 = gp.models.VariationalSparseGP(
            X=torch.empty(0, input_dim).to(device),
            y=None,
            kernel=gp.kernels.RBF(input_dim),
            Xu=torch.randn(num_inducing, input_dim).to(device),
            likelihood=None
        )

        # Projection MLP: 1D → hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(self.latent_dim1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)

        self.latent_dim2 = 16
        # Example: gp2 outputs [N, 16] latent vector
        self.gp2 = gp.models.VariationalSparseGP(
            X=torch.empty(0, hidden_dim).to(device),
            y=None,
            kernel=gp.kernels.RBF(hidden_dim),
            Xu=torch.randn(num_inducing, hidden_dim).to(device),
            likelihood=None
        )

        # 2nd projection: hidden_dim → hidden_dim
        self.projection2 = nn.Sequential(
            nn.Linear(self.latent_dim2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)

        # 3rd GP: hidden_dim → uncertainty
        self.gp3 = gp.models.VariationalSparseGP(
            X=torch.empty(0, hidden_dim).to(device),
            y=None,
            kernel=gp.kernels.RBF(hidden_dim),
            Xu=torch.randn(num_inducing, hidden_dim).to(device),
            likelihood=gp.likelihoods.Gaussian()
        )

    def forward(self, x):
        h = self.gp1.model(x)
        out = self.gp2.model(h)
        return out
    
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
    
    def train_vdgp(self, X_train, y_train, object_center, num_steps=500, lr=1e-4):
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        object_center = object_center.to(self.device)

        for model in [self.gp1, self.gp2, self.gp3]:
            model.Xu = model.Xu.to(self.device)
            for param in model.kernel.parameters():
                param.data = param.data.to(self.device)

        # Construct 5D input: [x, y, z, look_x, look_y]
        with torch.no_grad():
            look_dirs = F.normalize(object_center.unsqueeze(0) - X_train, dim=-1)  # [N, 3]
            X_train_5d = torch.cat([X_train, look_dirs[:, :2]], dim=-1)  # [N, 5]
            self.gp1.set_data(X=X_train_5d)
            h1_list = []
            for _ in range(self.latent_dim1):
                h1, _ = self.gp1.forward(X_train_5d)
                h1_list.append(h1.unsqueeze(-1))
            h1_mean = torch.cat(h1_list, dim=-1)  # [N, latent_dim1]
            h2_input = self.projection(h1_mean)   # [N, hidden_dim]

        with torch.no_grad():
            self.gp2.set_data(X=h2_input)
            h2_list = []
            for _ in range(self.latent_dim2):
                h2, _ = self.gp2.forward(h2_input)
                h2_list.append(h2.unsqueeze(-1))
            h2_mean = torch.cat(h2_list, dim=-1)  # [N, latent_dim2]
            h3_input = self.projection2(h2_mean).detach()  # [N, hidden_dim]

        self.gp3.set_data(X=h3_input, y=y_train)
        self.gp3.num_data = y_train.size(0)

        optimizer = pyro.optim.Adam({"lr": lr})
        elbo = pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(self.gp3.model, self.gp3.guide, optimizer, elbo)

        for i in range(num_steps):
            loss = svi.step()
            if (i + 1) % 50 == 0:
                print(f"Step {i+1}/{num_steps}, Loss: {loss:.3f}")


    def optimize_gp_posterior_vdgp(self, proposal_uvs, proposal_centers, uncertainties, init_uv, uv_bounds, radius, object_center, steps=100, lr=1e-2, beta=2.0):
        device = self.device
        u_min, u_max = uv_bounds[0]
        v_min, v_max = uv_bounds[1]

        # Normalize uncertainties
        y_train = uncertainties.to(device).squeeze()
        y_train = (y_train - y_train.mean()) / (y_train.std() + 1e-6)

        X_train = torch.tensor(np.array(proposal_centers), dtype=torch.float32, device=device)

        self.train_vdgp(X_train, y_train, object_center)

        u = torch.tensor([init_uv[0]], device=device, dtype=torch.float32, requires_grad=True)
        v = torch.tensor([init_uv[1]], device=device, dtype=torch.float32, requires_grad=True)
        uv_optimizer = torch.optim.Adam([u, v], lr=lr)

        for _ in range(steps):
            uv_optimizer.zero_grad()
            cam_center = uv2car_torch(u, v) * radius  # (1, 3)

            # Construct 5D input: [x, y, z, look_x, look_y]
            look_dir = F.normalize(object_center - cam_center.squeeze(0), dim=-1)  # (3,)
            cam_input = torch.cat([cam_center, look_dir[:2].unsqueeze(0)], dim=-1)  # (1, 5)

            with torch.no_grad():
                # GP1 forward
                h1_mean_list = []
                for _ in range(self.latent_dim1):
                    h1_mean, _ = self.gp1.forward(cam_input)
                    h1_mean_list.append(h1_mean.unsqueeze(-1))
                h1_mean = torch.cat(h1_mean_list, dim=-1)  # [1, latent_dim1]

                h2_input = self.projection(h1_mean)  # [1, hidden_dim]

                # GP2 forward
                h2_mean_list = []
                for _ in range(self.latent_dim2):
                    h2_mean, _ = self.gp2.forward(h2_input)
                    h2_mean_list.append(h2_mean.unsqueeze(-1))
                h2_mean = torch.cat(h2_mean_list, dim=-1)  # [1, latent_dim2]

                h3_input = self.projection2(h2_mean)  # [1, hidden_dim]

            if h3_input.ndim == 1:
                h3_input = h3_input.unsqueeze(0)

            mean, var = self.gp3.forward(h3_input, full_cov=False)
            acquisition = mean + beta * var.sqrt()
            loss = -acquisition
            loss.backward()
            uv_optimizer.step()

            u.data.clamp_(u_min, u_max)
            v.data.clamp_(v_min, v_max)

        final_uv = (u.item(), v.item())
        final_center = uv2car_torch(u.detach(), v.detach()).squeeze(0) * radius
        return final_center, final_uv


class GPFeatureExtractor(torch.nn.Sequential):
    def __init__(self, input_dim):
        super().__init__(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        )

class DeepGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        x = self.feature_extractor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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

        #Deep GP:
        if args.deepkgp:
            self.feature_extractor = GPFeatureExtractor(input_dim=3).to(self.device)
            self.likelihood = GaussianLikelihood().to(self.device)
            self.model = None  # will be set at training time
            self.ucb_beta = 2.0

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx: List[str] = [name2idx[k] for k in args.filter_out_grad]

    def train_dkl_gp(self, X_train, y_train, steps=200):
        X_train = X_train.to(self.device)
        y_train = y_train.squeeze().to(self.device)

        self.model = DeepGPModel(X_train, y_train, self.likelihood, self.feature_extractor).to(self.device)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(steps):  # Tune num iterations as needed
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

    def rbf_kernel(self, X1, X2):
        """
        Computes the RBF (squared exponential) kernel matrix using PyTorch.
        X1: (N, D) torch tensor
        X2: (M, D) torch tensor
        Returns: (N, M) torch tensor
        """
        # Compute squared Euclidean distance
        X1_sq = (X1 ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        X2_sq = (X2 ** 2).sum(dim=1, keepdim=True)  # (M, 1)
        dists = X1_sq - 2 * X1 @ X2.T + X2_sq.T     # (N, M)

        return self.sigma_f ** 2 * torch.exp(-0.5 / self.l**2 * dists)

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

    def optimize_gp_posterior(self, proposal_uvs, proposal_centers, uncertainties, init_uv, uv_bounds, radius, steps=200, lr=1e-2):
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

        # Optimize in uv-space
        u = torch.tensor([init_uv[0]], device=device, dtype=torch.float32, requires_grad=True)
        v = torch.tensor([init_uv[1]], device=device, dtype=torch.float32, requires_grad=True)

        # Prepare training data
        X_train = torch.tensor(np.array(proposal_centers), dtype=torch.float32, device=device)  # (N, 3)
        y_train = uncertainties.to(device).unsqueeze(1)  # (N, 1)

        # Compute kernel matrix K and its inverse (detached!)
        with torch.no_grad():
            K = self.rbf_kernel(X_train, X_train) + self.noise * torch.eye(X_train.size(0), device=device)
            K_inv = torch.inverse(K)  # (N, N)

        optimizer = torch.optim.Adam([u, v], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()

            cam_center = uv2car_torch(u, v) * radius  # (1, 3)
            K_s = self.rbf_kernel(X_train, cam_center)  # (N, 1)

            mu = K_s.T @ K_inv @ y_train  # (1, 1), differentiable w.r.t. u and v
            loss = -mu  # maximize posterior mean
            loss.backward()
            optimizer.step()

            u.data.clamp_(u_min, u_max)
            v.data.clamp_(v_min, v_max)

        final_uv = (u.item(), v.item())
        final_center = uv2car_torch(u.detach(), v.detach()).squeeze(0) * radius
        return final_center, final_uv

    def optimize_gp_posterior_dkl(self, proposal_uvs, proposal_centers, uncertainties, init_uv, uv_bounds, radius, steps=100, lr=1e-2):
        """
        Optimizes the GP posterior mean over the hemisphere using Deep Kernel Learning (DKL).
        Args:
            proposal_uvs: (N, 2) tensor of proposal (u, v) angles
            proposal_centers: (N, 3) tensor of world camera centers (training inputs)
            uncertainties: (N,) tensor of target uncertainties
            init_uv: (u, v) tuple - starting point for optimization
            uv_bounds: ((u_min, u_max), (v_min, v_max)) bounds for search
            radius: float - fixed distance to project camera center
            steps: int - number of optimization steps
            lr: float - learning rate
        Returns:
            optimized_cam_center: (3,) tensor
            optimized_uv: (u, v) tuple
        """
        device = self.device
        u_min, u_max = uv_bounds[0]
        v_min, v_max = uv_bounds[1]

        # Create training data tensors
        X_train = torch.tensor(proposal_centers, dtype=torch.float32, device=device)
        y_train = uncertainties.to(device).squeeze()  # (N,)

        # Train the DKL GP model
        self.train_dkl_gp(X_train, y_train)  # sets self.model and self.likelihood

        # Initialize UV parameters to optimize
        u = torch.tensor([init_uv[0]], device=device, dtype=torch.float32, requires_grad=True)
        v = torch.tensor([init_uv[1]], device=device, dtype=torch.float32, requires_grad=True)
        uv_optimizer = torch.optim.Adam([u, v], lr=lr)

        self.model.eval()
        self.likelihood.eval()

        for _ in range(steps):
            uv_optimizer.zero_grad()

            # Convert (u, v) to (x, y, z) camera center
            cam_center = uv2car_torch(u, v) * radius  # (1, 3)

            # Query GP prediction at this cam center
            with gpytorch.settings.fast_pred_var():
                pred = self.model(cam_center)
                mu = pred.mean  # (1,)
                sigma = pred.variance.sqrt()

            # Maximize the mean => minimize negative
            acquisition = mu + self.ucb_beta * sigma
            loss = -acquisition
            loss.backward()
            uv_optimizer.step()

            # Clamp UV values
            u.data.clamp_(u_min, u_max)
            v.data.clamp_(v_min, v_max)

        final_uv = (u.item(), v.item())
        final_center = uv2car_torch(u.detach(), v.detach()).squeeze(0) * radius
        return final_center, final_uv