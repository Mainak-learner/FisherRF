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
import torch.nn.functional as F

class VDGPFisherNBVSelector(Module):
    def __init__(self, args, input_dim, phi_pose_to_feat: nn.Module, hidden_dim=32, num_inducing=32, device="cuda"):
        super().__init__()
        self.device = device
        self.input_dim = input_dim  # pose_dim + halluc_feat_dim

        self.phi_pose_to_feat = phi_pose_to_feat.to(device)
        self.phi_pose_to_feat.eval()
        for p in self.phi_pose_to_feat.parameters():
            p.requires_grad = False

        self.hidden_dim = hidden_dim
        self.seed = args.seed
        self.reg_lambda = args.reg_lambda
        self.I_test = args.I_test
        self.I_acq_reg = args.I_acq_reg

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx = [name2idx[k] for k in args.filter_out_grad]

        self.latent_dim1 = 4

        self.gp1 = gp.models.VariationalSparseGP(
            X=torch.empty(0, input_dim).to(device),
            y=None,
            kernel=gp.kernels.RBF(input_dim),
            Xu=torch.randn(num_inducing, input_dim).to(device),
            likelihood=None,
        )

        self.projection = nn.Sequential(
            nn.Linear(self.latent_dim1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)

        self.gp2 = gp.models.VariationalSparseGP(
            X=torch.empty(0, hidden_dim).to(device),
            y=None,
            kernel=gp.kernels.RBF(hidden_dim),
            Xu=torch.randn(num_inducing, hidden_dim).to(device),
            likelihood=gp.likelihoods.Gaussian(),
        )

        # Freeze gp1 and allow training only on projection + gp2
        for param in self.gp1.parameters():
            param.requires_grad = False
        for param in self.projection.parameters():
            param.requires_grad = True
        for param in self.gp2.parameters():
            param.requires_grad = True

    def model(self):
        pyro.module("gp2", self.gp2)

        h1_list = [self.gp1(self.X_train_feat.detach())[0].unsqueeze(-1) for _ in range(self.latent_dim1)]
        h1_mean = torch.cat(h1_list, dim=-1)  # [N, latent_dim1]
        h2_input = self.projection(h1_mean)    # [N, hidden_dim]

        self.gp2.set_data(X=h2_input, y=self.y_train)
        return self.gp2.model()

    def guide(self):
        h1_list = [self.gp1(self.X_train_feat)[0].unsqueeze(-1) for _ in range(self.latent_dim1)]
        h1_mean = torch.cat(h1_list, dim=-1)
        h2_input = self.projection(h1_mean)

        self.gp2.set_data(X=h2_input, y=self.y_train)
        return self.gp2.guide()
    
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
        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)
        self.object_center = object_center.to(self.device)

        # Push GP models' Xu and kernel parameters to correct device
        for model in [self.gp1, self.gp2]:
            model.Xu = model.Xu.to(self.device)
            for param in model.kernel.parameters():
                param.data = param.data.to(self.device)

        with torch.no_grad():
            halluc_feats = self.phi_pose_to_feat(self.X_train).to(self.device)
            self.X_train_feat = torch.cat([self.X_train, halluc_feats], dim=-1)

        # Set inputs for GP1 (feature GP)
        self.gp1.set_data(X=self.X_train_feat, y=None)

        # Run gp1 → projection
        with torch.no_grad():
            h1_list = [self.gp1(self.X_train_feat)[0].unsqueeze(-1) for _ in range(self.latent_dim1)]
            h1_mean = torch.cat(h1_list, dim=-1)
            h2_input = self.projection(h1_mean)

        # Set inputs for gp2
        self.gp2.set_data(X=h2_input, y=self.y_train)
        self.gp2.num_data = self.y_train.size(0)

        optimizer = pyro.optim.Adam({"lr": lr})
        svi = pyro.infer.SVI(model=self.model, guide=self.guide, optim=optimizer, loss=pyro.infer.Trace_ELBO())

        for i in range(num_steps):
            loss = svi.step()
            if (i + 1) % 50 == 0:
                print(f"Step {i+1}/{num_steps}, Loss: {loss:.3f}")

    def optimize_gp_posterior_vdgp(self, proposal_uvs, proposal_centers, uncertainties,
                                init_uv, uv_bounds, radius, object_center,
                                steps=100, lr=1e-2, beta=2.0):
        device = self.device
        u_min, u_max = uv_bounds[0]
        v_min, v_max = uv_bounds[1]

        # Normalize uncertainties
        y_train = uncertainties.to(device).squeeze()
        y_train = (y_train - y_train.mean()) / (y_train.std() + 1e-6)

        # Train GP
        X_train = torch.tensor(np.array(proposal_centers), dtype=torch.float32, device=device)
        self.train_vdgp(X_train, y_train, object_center)

        # Optimize pose (u, v)
        u = torch.tensor([init_uv[0]], dtype=torch.float32, device=device, requires_grad=True)
        v = torch.tensor([init_uv[1]], dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([u, v], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            cam_center = uv2car_torch(u, v) * radius  # (1, 3)

            with torch.no_grad():
                halluc_feat = self.phi_pose_to_feat(cam_center)
                cam_feat = torch.cat([cam_center, halluc_feat], dim=-1)

            # Inference through frozen gp1 → projection → gp2
            h1_list = [self.gp1(cam_feat)[0].unsqueeze(-1) for _ in range(self.latent_dim1)]
            h1_mean = torch.cat(h1_list, dim=-1)
            h2_input = self.projection(h1_mean)

            mean, var = self.gp2(h2_input, full_cov=False)
            acquisition = mean + beta * var.sqrt()
            loss = -acquisition
            loss.backward()
            optimizer.step()

            u.data.clamp_(u_min, u_max)
            v.data.clamp_(v_min, v_max)

        final_uv = (u.item(), v.item())
        final_center = uv2car_torch(u.detach(), v.detach()).squeeze(0) * radius
        return final_center, final_uv

class ThinPlateSpline2DKernel(torch.nn.Module):
    """
    Implements the closed-form 2D thin-plate spline covariance function:
        c(r) = 2r^2 log|r| - (1 + 2 log(R))r^2 + R^2
    where:
        r = ||x - x'||
        R = max pairwise distance (controls regularization)
    """
    def __init__(self):
        super().__init__()
        self.eps = 1e-6  # stability to avoid log(0)

    def forward(self, x1, x2):
        x1 = x1 if x1.ndim == 2 else x1.view(-1, x1.size(-1))
        x2 = x2 if x2.ndim == 2 else x2.view(-1, x2.size(-1))

        dists = torch.cdist(x1, x2) + self.eps  # shape (N, M)
        R = torch.max(dists).detach()  # scalar

        term1 = 2 * (dists ** 2) * torch.log(dists)
        term2 = (1 + 2 * torch.log(R)) * (dists ** 2)
        term3 = R ** 2

        return term1 - term2 + term3

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

        if kernel_type == "tps":
            self.base_kernel = ThinPlateSpline2DKernel()
        elif kernel_type == "rbf":
            self.base_kernel = RBFKernel()
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        self.covar_module = ScaleKernel(lambda x1, x2: self.base_kernel(x1, x2)) if kernel_type == "tps" else ScaleKernel(self.base_kernel)

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
            self.feature_extractor = GPFeatureExtractor(input_dim=3, output_dim=out_dim).to(self.device)            
            self.likelihood = GaussianLikelihood().to(self.device)
            self.model = None  # will be set at training time
            self.ucb_beta = 2.0

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx: List[str] = [name2idx[k] for k in args.filter_out_grad]

    def compute_pose_overlap_score(self, new_center, acquired_cameras, sim_tau=0.5):
        """
        Differentiable overlap score encouraging NBV proposals to share view overlap with past views.
        """
        if not acquired_cameras:
            return torch.tensor(0.0, device=new_center.device)

        distances = []
        for cam in acquired_cameras:
            center_q = cam.camera_center.detach()
            dist = torch.norm(new_center - center_q, p=2)
            distances.append(torch.exp(-dist / sim_tau))

        score_tensor = torch.stack(distances)
        return torch.logsumexp(score_tensor, dim=0) - np.log(len(distances))

    def train_dkl_gp(self, X_train, y_train, steps=200):
        X_train = X_train.to(self.device)
        y_train = y_train.squeeze().to(self.device)

        self.model = DeepTPSGPModel(X_train, y_train, self.likelihood, self.feature_extractor, kernel_type=self.kernel_type).to(self.device)       
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
    
    def reparameterized_acquisition(self, mu, sigma, num_samples=10, beta=1.0):
        """
        Reparameterized sampling of acquisition values.
        """
        eps = torch.randn((num_samples,) + mu.shape, device=mu.device)  # (S, B)
        samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps            # (S, B)

        # Option 1: UCB-like sampling (stochastic upper confidence bound)
        # sampled_acq = samples + beta * sigma.unsqueeze(0)               # shape (S, B)

        # Option 2: just use sampled value directly
        sampled_acq = samples

        return sampled_acq.mean(dim=0)

    def optimize_gp_posterior_dkl(
        self,
        proposal_uvs,
        proposal_centers,
        uncertainties,
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

        self.train_dkl_gp(X_train, y_train)
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

        for _ in range(steps):
            uv_optimizer.zero_grad()

            cam_center = uv2car_torch(u, v) * radius  # (1, 3)

            with gpytorch.settings.fast_pred_var():
                pred = self.model(cam_center)
                mu = pred.mean
                sigma = pred.variance.sqrt()

            # Overlap score based on pose proximity
            overlap_score = self.compute_pose_overlap_score(cam_center.squeeze(0), selected_cameras)

            # Weighted acquisition function
            acq = self.reparameterized_acquisition(mu, sigma, num_samples=1, beta=1.0)
            loss = -acq.sum()  # maximize acquisition => minimize negative
            loss.backward()

            uv_optimizer.step()

            u.data.clamp_(u_min, u_max)
            v.data.clamp_(v_min, v_max)

        final_uv = (u.item(), v.item())
        final_center = uv2car_torch(u.detach(), v.detach()).squeeze(0) * radius
        return final_center, final_uv