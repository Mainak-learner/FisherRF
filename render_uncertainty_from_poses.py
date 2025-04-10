import torch
import os
import numpy as np
from scene import Scene, GaussianModel
from gaussian_renderer import render, forward_k_times, modified_render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
import torchvision
import json
from einops import reduce, repeat
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Define load_model function
def load_model(checkpoint_path, dataset, opt):
    gaussians = GaussianModel(dataset, is_variational=True)
    # Explicitly set dataset_name if not provided
    if not hasattr(dataset, 'dataset_name') or dataset.dataset_name is None:
        dataset.dataset_name = "BLENDER"  # Default to BLENDER; adjust based on your dataset
    # Ensure source_path is set
    if not hasattr(dataset, 'source_path') or dataset.source_path == "":
        dataset.source_path = os.path.dirname(checkpoint_path)
    scene = Scene(dataset, gaussians)
    if os.path.exists(checkpoint_path):
        ckpt_dict = torch.load(checkpoint_path, weights_only=False)
        gaussians.restore(ckpt_dict["model_params"], opt)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return gaussians, scene

def precompute_H_per_gaussian(gaussians, scene, pipeline, background):
    # Use train and test views to compute H_per_gaussian
    train_views = scene.getTrainCameras()
    test_views = scene.getTestCameras()
    all_views = train_views + test_views

    params = [gaussians._xyz, gaussians._features_dc, gaussians._features_rest, gaussians._scaling, gaussians._opacity]
    params = [p.requires_grad_(True) for p in params if p.grad is None or not any(k in ["rotation"] for k in ["rotation"])]
    optim = torch.optim.SGD(params, 0.)
    gaussians.optimizer = optim
    device = params[0].device

    H_per_gaussian = torch.zeros(params[0].shape[0], device=device, dtype=params[0].dtype)

    for view in tqdm(all_views, desc="Precomputing H_per_gaussian"):
        render_pkg = modified_render(view, gaussians, pipeline, background)
        pred_img = render_pkg["render"]
        pred_img.backward(gradient=torch.ones_like(pred_img))
        H_per_gaussian += sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in params])
        optim.zero_grad(set_to_none=True)

    return H_per_gaussian.detach()

def render_uncertainty_from_test_views(scene, gaussians, pipe, background, output_dir, H_per_gaussian, num_views=5):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all test views
    test_views = scene.getTestCameras()
    print(f"Total test views available: {len(test_views)}")
    
    # Randomly select num_views from test views
    if len(test_views) <= num_views:
        selected_views = test_views
    else:
        selected_indices = random.sample(range(len(test_views)), num_views)
        selected_views = [test_views[i] for i in selected_indices]
    
    print(f"Selected {len(selected_views)} test views")
    
    for idx, viewpoint in enumerate(selected_views):
        print(f"Processing test view {idx}: {viewpoint.image_name}")
        
        # Variational Uncertainty
        variational_pkg = forward_k_times(viewpoint, gaussians, pipe, background, k=gaussians.n_models)
        var_rgb = variational_pkg["comp_rgb"].detach()
        var_uncertainty = variational_pkg["comp_std"].detach()
        # Convert to single-channel uncertainty
        var_uncertainty = var_uncertainty.mean(dim=0, keepdim=True)  # Shape: (1, height, width)

        # FisherRF Uncertainty with precomputed H_per_gaussian
        xyz = gaussians._xyz
        to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)], dim=-1)
        pts3d_homo = to_homo(xyz)
        pts3d_cam = pts3d_homo @ viewpoint.world_view_transform
        gaussian_depths = pts3d_cam[:, 2, None].clamp(min=0)

        hessian_color = repeat(H_per_gaussian, "n -> n c", c=3)
        cur_hessian_color = hessian_color * gaussian_depths
        fisher_render_pkg = render(viewpoint, gaussians, pipe, background, override_color=cur_hessian_color)
        fisher_rgb = fisher_render_pkg["render"]
        
        render_pkg = modified_render(viewpoint, gaussians, pipe, background)
        pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
        fisher_uncertainty = reduce(fisher_render_pkg["render"], "c h w -> h w", "mean").detach()
        fisher_uncertainty = torch.log(fisher_uncertainty / pixel_gaussian_counter.clamp(min=1e-6))

        # Ground truth image
        gt_image = viewpoint.original_image.cuda() if viewpoint.original_image is not None else None

        # Normalize uncertainties to 0-1
        var_min, var_max = var_uncertainty.min(), var_uncertainty.max()
        var_uncertainty_norm = (var_uncertainty - var_min) / (var_max - var_min + 1e-6)

        fisher_min, fisher_max = fisher_uncertainty.min(), fisher_uncertainty.max()
        fisher_uncertainty_norm = (fisher_uncertainty - fisher_min) / (fisher_max - fisher_min + 1e-6)

        # Ensure both uncertainties have the same number of dimensions
        var_uncertainty_norm = var_uncertainty_norm.squeeze(0)  # Remove singleton channel dimension
        fisher_uncertainty_norm = fisher_uncertainty_norm.unsqueeze(0)  # Add singleton channel dimension

        # Convert tensors to numpy for plotting
        render_img_np = var_rgb.cpu().numpy().transpose(1, 2, 0)  # RGB image: (height, width, 3)
        fisher_unc_np = fisher_uncertainty_norm.squeeze(0).cpu().numpy()  # Single-channel: (height, width)
        var_unc_np = var_uncertainty_norm.cpu().numpy()  # Single-channel: (height, width)
        
        # Create subplots
        if gt_image is not None:
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            gt_np = gt_image.cpu().numpy().transpose(1, 2, 0)
            axs[0].imshow(gt_np)
            axs[0].set_title(f"Ground Truth - View {viewpoint.image_name}")
            axs[0].axis('off')
            
            axs[1].imshow(render_img_np)
            axs[1].set_title(f"Render - View {viewpoint.image_name}")
            axs[1].axis('off')
            
            axs[2].imshow(fisher_unc_np, cmap='viridis')
            axs[2].set_title(f"FisherRF Uncertainty")
            axs[2].axis('off')
            
            axs[3].imshow(var_unc_np, cmap='viridis')
            axs[3].set_title(f"Variational Uncertainty")
            axs[3].axis('off')
        else:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(render_img_np)
            axs[0].set_title(f"Render - View {viewpoint.image_name}")
            axs[0].axis('off')
            
            axs[1].imshow(fisher_unc_np, cmap='viridis')
            axs[1].set_title(f"FisherRF Uncertainty")
            axs[1].axis('off')
            
            axs[2].imshow(var_unc_np, cmap='viridis')
            axs[2].set_title(f"Variational Uncertainty")
            axs[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{viewpoint.image_name}.png"))
        plt.show()
        plt.close()

        # Save images separately
        # torchvision.utils.save_image(var_rgb, os.path.join(output_dir, f"var_rgb_{viewpoint.image_name}.png"))
        # torchvision.utils.save_image(fisher_rgb, os.path.join(output_dir, f"fisher_rgb_{viewpoint.image_name}.png"))
        
        # # Save side-by-side comparison
        # side_by_side_rgb = torch.cat([var_rgb, fisher_rgb], dim=2)  # Concatenate along width
        # side_by_side_unc = torch.cat([var_uncertainty_norm, fisher_uncertainty_norm], dim=2)  # Concatenate along width
        # torchvision.utils.save_image(side_by_side_rgb, os.path.join(output_dir, f"rgb_comparison_{viewpoint.image_name}.png"))
        # torchvision.utils.save_image(side_by_side_unc, os.path.join(output_dir, f"uncertainty_comparison_{viewpoint.image_name}.png"))

if __name__ == "__main__":
    parser = ArgumentParser(description="Render uncertainty from test views")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="./uncertainty_renders", help="Output directory for renders")
    parser.add_argument("--num_views", type=int, default=5, help="Number of test views to render")

    args = parser.parse_args()

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    gaussians, scene = load_model(args.checkpoint, dataset, opt)

    # Define the background color before calling precompute_H_per_gaussian
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Precompute H_per_gaussian using training and testing views
    H_per_gaussian = precompute_H_per_gaussian(gaussians, scene, pipe, background)

    # Render uncertainty from random test views
    render_uncertainty_from_test_views(scene, gaussians, pipe, background, args.output_dir, H_per_gaussian, args.num_views)
    print(f"Uncertainty renders and plots saved to {args.output_dir}")