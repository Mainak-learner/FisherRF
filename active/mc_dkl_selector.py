import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam
import torch
import numpy as np
from tqdm import tqdm
from utils.graphics_utils import uv2car_torch
from gaussian_renderer import render, network_gui, modified_render



class DropoutPoseEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[128, 64], output_dim=64, dropout_rate=0.1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MCDKLGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, encoder):
        super().__init__(train_x, train_y, likelihood)
        self.encoder = encoder
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        x_feat = self.encoder(x)
        mean_x = self.mean_module(x_feat)
        covar_x = self.covar_module(x_feat)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MCDKLNBVSelector(nn.Module):
    def __init__(self, args, dropout_rate=0.1, beta=2.0, device="cuda"):
        super().__init__()
        self.device = device
        self.beta = beta
        self.encoder = DropoutPoseEncoder(dropout_rate=dropout_rate).to(device)
        self.likelihood = GaussianLikelihood().to(device)
        self.model = None
        self.grad_reg_lambda = 0.3

        self.seed = args.seed
        self.reg_lambda = args.reg_lambda
        self.I_test: bool = args.I_test
        self.I_acq_reg: bool = args.I_acq_reg

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx: List[str] = [name2idx[k] for k in args.filter_out_grad]

    def train_gp(self, X_train, y_train, steps=200, lr=1e-3):
        self.model = MCDKLGPModel(X_train, y_train, self.likelihood, self.encoder).to(self.device)
        self.model.train()
        self.likelihood.train()
        optimizer = Adam(self.model.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        for _ in range(steps):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

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

    def predict_with_uncertainty(self, x_test, T=100):
        self.model.eval()
        self.likelihood.eval()
        preds = []
        with gpytorch.settings.fast_pred_var():  # ← Keep only this
            for _ in range(T):
                pred = self.likelihood(self.model(x_test))
                preds.append(pred.mean.unsqueeze(0))
        means = torch.cat(preds, dim=0)
        mean = means.mean(dim=0)
        std = means.std(dim=0)
        return mean, std

    def optimize_gp_posterior_dkl(
        self,
        proposal_centers,  # List[(3,)]
        uncertainties,     # Tensor (N,)
        init_uv,           # Tuple[float, float]
        uv_bounds,         # Tuple[(u_min, u_max), (v_min, v_max)]
        radius,            # float
        steps=100,
        lr=1e-2
    ):
        """
        Args:
            proposal_centers: List of world 3D camera positions (N, 3)
            uncertainties: Uncertainty values at proposal poses (N,)
            init_uv: Initial (u, v)
            uv_bounds: ((u_min, u_max), (v_min, v_max))
            radius: Radius to convert UV to 3D
        Returns:
            final_center: Optimized 3D camera center
            final_uv: Optimized (u, v) pair
        """
        device = self.device
        u_min, u_max = uv_bounds[0]
        v_min, v_max = uv_bounds[1]

        # Prepare GP training data
        X_train = torch.stack([pc.detach() if pc.requires_grad else pc for pc in proposal_centers]).float().to(device)
        y_train = uncertainties.to(device).squeeze()

        self.train_gp(X_train, y_train)

        # Init optimization parameters
        u = torch.tensor([init_uv[0]], requires_grad=True, dtype=torch.float32, device=device)
        v = torch.tensor([init_uv[1]], requires_grad=True, dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam([u, v], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            cam_center = uv2car_torch(u, v) * radius  # (1, 3)
            cam_center.requires_grad_(True)

            mu, sigma = self.predict_with_uncertainty(cam_center)
            acquisition = mu + self.beta * sigma
            loss = -acquisition.mean()

            # Gradient regularization
            grad = torch.autograd.grad(acquisition.mean(), cam_center, create_graph=True)[0]
            loss += self.grad_reg_lambda * grad.detach().norm()**2

            loss.backward()
            optimizer.step()

            u.data.clamp_(u_min, u_max)
            v.data.clamp_(v_min, v_max)

        final_uv = (u.item(), v.item())
        final_center = uv2car_torch(u.detach(), v.detach()).squeeze(0) * radius
        return final_center, final_uv
