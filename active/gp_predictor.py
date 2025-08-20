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
from gpytorch.kernels import Kernel
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
import torch.nn.functional as F

def make_kernel(kernel_type: str, embedding_dim: int = None):
    kernel_type = kernel_type.lower()
    if kernel_type == "rbf":
        return gpytorch.kernels.RBFKernel()
    elif kernel_type == "matern":
        return gpytorch.kernels.MaternKernel(nu=2.5)
    elif kernel_type == "rq":
        return gpytorch.kernels.RationalQuadraticKernel()
    elif kernel_type == "linear":
        return gpytorch.kernels.LinearKernel()
    elif kernel_type == "periodic":
        return gpytorch.kernels.PeriodicKernel()
    elif kernel_type == "spectral":
        return gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=4, ard_num_dims=embedding_dim
        )
    elif kernel_type == "rbf+linear":
        return gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel()
    elif kernel_type == "matern+periodic":
        return gpytorch.kernels.MaternKernel(nu=2.5) + gpytorch.kernels.PeriodicKernel()
    elif kernel_type == "rbf*periodic":
        return gpytorch.kernels.RBFKernel() * gpytorch.kernels.PeriodicKernel()
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

class GPFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  # Output: meaningful 2D coordinates
        )

    def forward(self, x):
        out = self.net(x)
        return F.normalize(out, dim=-1)  # Encourage outputs on unit circle

class DeepTPSGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, kernel_type="tps"):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel_type = kernel_type

        # Pick kernel based on args
        self.base_covar = make_kernel(kernel_type, embedding_dim=feature_extractor.output_dim)

        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_covar)

    def forward(self, x):
        x_feat = self.feature_extractor(x)
        mean_x = self.mean_module(x_feat)
        covar_x = self.covar_module(x_feat, x_feat)
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
        self.sim_lambda = 0.3
        self.kernel_type = getattr(args, "kernel_type", "rbf")  # default to 'rbf'

        #Deep GP:
        if args.deepkgp:
            out_dim = 2 if self.kernel_type == "tps" else 64
            if args.exclude_deep_kernel:
                self.feature_extractor = torch.nn.Identity()
            else:
                self.feature_extractor = GPFeatureExtractor(input_dim=3, output_dim=out_dim).to(self.device)

            self.likelihood = GaussianLikelihood().to(self.device)
            self.model = None  # will be set at training time
            self.ucb_beta = 2.0

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx: List[str] = [name2idx[k] for k in args.filter_out_grad]

    @torch.no_grad()
    def sample_view_manifold(self, num_samples, u_bounds, v_bounds, radius, device=None):
        if device is None:
            device = self.device
        u_min, u_max = u_bounds
        v_min, v_max = v_bounds
        us = torch.empty(num_samples, device=device).uniform_(u_min, u_max)
        vs = torch.empty(num_samples, device=device).uniform_(v_min, v_max)
        centers = uv2car_torch(us, vs) * radius  # (num_samples, 3) on `device`
        return centers

    
    @torch.no_grad()
    def estimate_variance_statistics(self, model, cam_centers):
        # Use the same device as the model’s training inputs
        model_device = model.train_inputs[0].device
        cam_centers = cam_centers.to(model_device).float()

        model.eval(); self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            preds = model(cam_centers)
            sigma = preds.variance.sqrt()

        sigma_mean = sigma.mean().item()
        sigma_max  = sigma.max().item()
        return sigma_mean, sigma_max

    def train_dkl_gp(self, X_train, y_train, kernel_type = "rbf", steps=200):
        X_train = X_train.to(self.device)
        y_train = y_train.squeeze().to(self.device)

        self.model = DeepTPSGPModel(X_train, y_train, self.likelihood, self.feature_extractor, kernel_type=kernel_type).to(self.device)       
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

    def compute_fisher_uncertainty(self, gaussians, candidate_cameras, I_train_diag, pipe, background):
        params = gaussians.capture()[1:7]
        params = [p for i, p in enumerate(params) if i not in self.filter_out_idx]
        device = params[0].device

        acq_scores = torch.zeros(len(candidate_cameras), device=device)

        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating Hessian on proposal views")):
            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])
            if self.I_acq_reg:
                cur_H += self.reg_lambda

            gaussians.optimizer.zero_grad(set_to_none=True)
            acq_scores[idx] = torch.sum(cur_H * I_train_diag).item()

        return acq_scores
    
    def reparameterized_acquisition(self, mu, sigma, num_samples, beta=1.0):
        """
        Reparameterized sampling of acquisition values.
        """
        # eps = torch.randn((num_samples,) + mu.shape, device=mu.device)  # (S, B)
        # samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps            # (S, B)

        # Option 1: UCB (upper confidence bound)
        sampled_acq = mu + beta * sigma               # shape (S)

        # Option 2: just use sampled value directly
        # sampled_acq = samples

        return sampled_acq

    def optimize_gp_posterior_dkl(
        self,
        proposal_uvs,
        proposal_centers,
        uncertainties,
        dense_centers,
        dense_uvs,
        init_uv,
        uv_bounds,
        radius,
        object_center,
        selected_cameras,
        gaussians,
        pipe,
        background,
        reference_camera,
        render_fn,
        image_encoder,
        steps=100,
        lr=1e-2,
        kernel_type = "rbf"
    ):
        """
        Optimizes GP posterior mean over the hemisphere using Deep Kernel Learning,
        with cosine similarity to past views included.

        Args:
            proposal_uvs: list of (u, v) tuples
            proposal_centers: list of (3,) world camera centers
            uncertainties: (N,) tensor of uncertainty values
            init_uv: (u, v) starting point
            uv_bounds: ((u_min, u_max), (v_min, v_max))
            radius: float camera radius
            object_center: (3,) tensor
            selected_cameras: list of DummyCamera with .original_image
            gaussians, pipe, background, reference_camera: rendering components
            image_encoder: instance of ImageEncoder
            steps, lr: optimization parameters
        Returns:
            optimized_cam_center: (3,) tensor
            optimized_uv: (u, v) tuple
        """
        device = self.device
        u_min, u_max = uv_bounds[0]
        v_min, v_max = uv_bounds[1]

        # Prepare GP training data
        X_train = torch.tensor(proposal_centers, dtype=torch.float32, device=device)
        y_train = uncertainties.to(device).squeeze()

        self.train_dkl_gp(X_train, y_train, kernel_type)
        self.model.eval()
        self.likelihood.eval()
        image_encoder.eval()

        # Optimize in uv-space
        u = torch.tensor([init_uv[0]], device=device, dtype=torch.float32, requires_grad=True)
        v = torch.tensor([init_uv[1]], device=device, dtype=torch.float32, requires_grad=True)
        uv_optimizer = torch.optim.Adam([u, v], lr=lr)

        # Prepare features from selected views
        ref_imgs = [F.interpolate(cam.original_image.unsqueeze(0), size=(224, 224)) for cam in selected_cameras]
        ref_imgs = torch.cat(ref_imgs, dim=0).to(device)  # (B, 3, 224, 224)
        with torch.no_grad():
            ref_feats = F.normalize(image_encoder(ref_imgs), dim=1)  # (B, D)

        cam_centers = self.sample_view_manifold(256, uv_bounds[0], uv_bounds[1], radius, device=self.device)
        cam_centers = cam_centers.to(self.model.train_inputs[0].device)
        sigma_mean, sigma_max = self.estimate_variance_statistics(self.model, cam_centers)
        beta = 1.5 * (sigma_max / max(sigma_mean, 1e-8))  # example adaptive beta

        dense_centers = torch.stack(dense_centers, dim=0)
        dense_centers = dense_centers.to(self.model.train_inputs[0].device)
        acq_dense = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(dense_centers.shape[0]):
                cam_center = dense_centers[i].unsqueeze(0).to(dtype=torch.float32)                
                pred = self.model(cam_center)
                mu = pred.mean
                sigma = pred.variance.sqrt()
                acq = self.reparameterized_acquisition(mu, sigma, num_samples=1, beta=beta)
                acq_dense.append(acq.item())

        acq_dense = np.array(acq_dense)


        for _ in range(steps):
            uv_optimizer.zero_grad()

            cam_center = uv2car_torch(u, v) * radius  # (1, 3)

            with gpytorch.settings.fast_pred_var():
                pred = self.model(cam_center)
                mu = pred.mean
                sigma = pred.variance.sqrt()

            # Weighted acquisition function
            acq = self.reparameterized_acquisition(mu, sigma, 1, beta)
            loss = -acq.sum()  # maximize acquisition => minimize negative
            loss.backward()

            uv_optimizer.step()

            u.data.clamp_(u_min, u_max)
            v.data.clamp_(v_min, v_max)

        final_uv = (u.item(), v.item())
        final_center = uv2car_torch(u.detach(), v.detach()).squeeze(0) * radius
        return final_center, final_uv, dense_uvs, acq_dense