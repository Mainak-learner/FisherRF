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


def capture(gaussians):
    """Captures the current state of the gaussians model."""
    return (
        gaussians.active_sh_degree,
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        gaussians.max_radii2D,
        gaussians.xyz_gradient_accum,
        gaussians.denom,
    )


def load_model(checkpoint_path, dataset, opt):
    """Load a trained Gaussian Splatting model from checkpoint."""
    # Initialize the Gaussian model - match the original code
    is_variational = True  # Set to True to match the second script
    gaussians = GaussianModel(dataset.sh_degree, is_variational=is_variational)
    
    # Ensure dataset parameters are properly set
    if not hasattr(dataset, 'dataset_name') or dataset.dataset_name is None:
        dataset.dataset_name = "BLENDER"
    
    if not hasattr(dataset, 'source_path') or dataset.source_path == "":
        dataset.source_path = os.path.dirname(checkpoint_path)
    
    # Create scene with simple parameters - matching original code
    iteration = opt.iterations if hasattr(opt, 'iterations') else -1
    print(f"Loading trained model at iteration {iteration}")
    
    # Simple scene creation matching the original code
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    # Now load the model checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt_dict = torch.load(checkpoint_path, weights_only=False)
        gaussians.restore(ckpt_dict["model_params"], opt)
        print(f"Loaded checkpoint successfully with {gaussians._xyz.shape[0]} gaussians")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Print camera counts for debugging
    print(f"Scene loaded with {len(scene.getTrainCameras())} train cameras and {len(scene.getTestCameras())} test cameras")
    
    return gaussians, scene


@torch.no_grad()
def render_uncertainty(view, gaussians, pipeline, background, hessian_color):
    """Render uncertainty using FisherRF method."""
    # Get standard render and pixel_gaussian_counter
    render_pkg = modified_render(view, gaussians, pipeline, background)
    pred_img = render_pkg["render"]
    pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
    
    # Render with hessian-based colors
    render_pkg = modified_render(view, gaussians, pipeline, background, override_color=hessian_color)
    
    # Calculate uncertainty map
    uncertainty_map = reduce(render_pkg["render"], "c h w -> h w", "mean")
    
    return pred_img, uncertainty_map, pixel_gaussian_counter, render_pkg["depth"]


def precompute_H_per_gaussian(gaussians, scene, pipeline, background):
    """Precompute the Hessian values per Gaussian for uncertainty estimation."""
    # Use both train and test views for better estimation
    train_views = scene.getTrainCameras()
    test_views = scene.getTestCameras()
    
    print(f"Using {len(train_views)} train views and {len(test_views)} test views for Hessian computation")
    
    # Get parameters to compute gradients for
    # Exclude rotation parameters (problematic for uncertainty)
    params = [
        gaussians._xyz, 
        gaussians._features_dc, 
        gaussians._features_rest, 
        gaussians._scaling,
        gaussians._opacity
    ]
    
    # Set up parameters for gradient computation
    params = [p.requires_grad_(True) for p in params]
    optim = torch.optim.SGD(params, 0.)
    gaussians.optimizer = optim
    device = params[0].device
    
    # Initialize tensor to accumulate Hessian diagonals
    H_per_gaussian = torch.zeros(gaussians._xyz.shape[0], device=device, dtype=params[0].dtype)
    
    # Only use a subset of views for computational efficiency if needed
    all_views = train_views + test_views
    if len(all_views) > 20:  # Limit to 20 views for faster computation
        selected_indices = np.linspace(0, len(all_views)-1, 20, dtype=int)
        computation_views = [all_views[i] for i in selected_indices]
        print(f"Using {len(computation_views)} views for Hessian computation")
    else:
        computation_views = all_views
    
    # Compute Hessian approximation
    for view in tqdm(computation_views, desc="Precomputing H_per_gaussian"):
        # Render image
        render_pkg = modified_render(view, gaussians, pipeline, background)
        pred_img = render_pkg["render"]
        
        # Compute gradient
        pred_img.backward(gradient=torch.ones_like(pred_img))
        
        # Accumulate gradient norms per Gaussian
        H_per_gaussian += sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in params])
        
        # Zero gradients for next iteration
        optim.zero_grad(set_to_none=True)
    
    # Normalize by number of views
    H_per_gaussian /= len(computation_views)
    
    return H_per_gaussian.detach()


def render_uncertainty_from_test_views(scene, gaussians, pipe, background, output_dir, H_per_gaussian, num_views=5):
    """Render and compare uncertainty estimates from both FisherRF and variational methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test views
    test_views = scene.getTestCameras()
    print(f"Total test views available: {len(test_views)}")
    
    # # If no test views are available, use a subset of train views
    # if len(test_views) == 0:
    #     print("No test views found. Using train views instead.")
    #     train_views = scene.getTrainCameras()
    #     print(f"Total train views available: {len(train_views)}")
        
    #     # Use train views as test views
    #     test_views = train_views
    
    # Get XYZ of gaussians for depth calculation
    xyz = gaussians._xyz
    
    # # Randomly select views if needed
    # if len(test_views) <= num_views:
    #     selected_views = test_views
    # else:
    #     selected_indices = random.sample(range(len(test_views)), num_views)
    #     selected_views = [test_views[i] for i in selected_indices]
    
    # print(f"Selected {len(selected_views)} views for uncertainty visualization")
    
    # Render each view
    for idx, viewpoint in enumerate(test_views):
        print(f"Processing test view {idx}: {viewpoint.image_name}")
        
        # Variational Uncertainty (if model is variational)
        if hasattr(gaussians, 'n_models') and gaussians.n_models > 1:
            variational_pkg = forward_k_times(viewpoint, gaussians, pipe, background, k=gaussians.n_models)
            var_rgb = variational_pkg["comp_rgb"].detach()
            var_uncertainty = variational_pkg["comp_std"].detach()
            var_uncertainty = var_uncertainty.mean(dim=0, keepdim=True)
        else:
            # If not variational, just use standard render
            var_pkg = render(viewpoint, gaussians, pipe, background)
            var_rgb = var_pkg["render"].detach()
            var_uncertainty = torch.zeros((1, var_rgb.shape[1], var_rgb.shape[2]), device=var_rgb.device)
        
        # FisherRF Uncertainty with precomputed H_per_gaussian
        # Convert points to homogeneous coordinates
        to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)], dim=-1)
        pts3d_homo = to_homo(xyz)
        
        # Transform to camera space
        pts3d_cam = pts3d_homo @ viewpoint.world_view_transform
        
        # Get depth values and ensure they are positive
        gaussian_depths = pts3d_cam[:, 2, None].clamp(min=0)
        
        # Create color tensor from Hessian values
        hessian_color = repeat(H_per_gaussian, "n -> n c", c=3)
        
        # Scale by depth for better visualization
        depth_weighted_hessian = hessian_color * gaussian_depths
        
        # Render with Hessian-weighted colors
        fisher_render_pkg = render(viewpoint, gaussians, pipe, background, override_color=depth_weighted_hessian)
        fisher_rgb = fisher_render_pkg["render"]
        
        # Get pixel Gaussian counter for normalization
        render_pkg = modified_render(viewpoint, gaussians, pipe, background)
        pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
        
        # Calculate Fisher uncertainty map
        fisher_uncertainty = reduce(fisher_render_pkg["render"], "c h w -> h w", "mean").detach()
        fisher_uncertainty = torch.log(fisher_uncertainty / pixel_gaussian_counter.clamp(min=1e-6))
        
        # Get ground truth image if available
        gt_image = viewpoint.original_image.cuda() if viewpoint.original_image is not None else None
        
        # Normalize uncertainties to 0-1 range for visualization
        var_min, var_max = var_uncertainty.min(), var_uncertainty.max()
        var_uncertainty_norm = (var_uncertainty - var_min) / (var_max - var_min + 1e-6)
        
        fisher_min, fisher_max = fisher_uncertainty.min(), fisher_uncertainty.max()
        fisher_uncertainty_norm = (fisher_uncertainty - fisher_min) / (fisher_max - fisher_min + 1e-6)
        
        # Match dimensions for consistent visualization
        var_uncertainty_norm = var_uncertainty_norm.squeeze(0)
        fisher_uncertainty_norm = fisher_uncertainty_norm.unsqueeze(0)
        
        # Convert to numpy for plotting
        render_img_np = var_rgb.cpu().numpy().transpose(1, 2, 0)
        fisher_unc_np = fisher_uncertainty_norm.squeeze(0).cpu().numpy()
        var_unc_np = var_uncertainty_norm.cpu().numpy()
        
        # Create visualization plots
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
        plt.close()
        
        # Save raw data for further analysis
        np.savez(os.path.join(output_dir, f"uncertainty_data_{viewpoint.image_name}.npz"), 
                 fisher_uncertainty=fisher_uncertainty.cpu().numpy(),
                 var_uncertainty=var_uncertainty.cpu().numpy() if hasattr(gaussians, 'n_models') else None,
                 pixel_gaussian_counter=pixel_gaussian_counter.cpu().numpy(),
                 depth=gaussian_depths.cpu().numpy()
                )


if __name__ == "__main__":
    # Set up command line argument parser - match the original code structure
    parser = ArgumentParser(description="Render uncertainty from test views")
    model = ModelParams(parser)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    
    # Add the same arguments as in the original render_uncertainty.py
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="./uncertainty_renders", help="Output directory for renders")
    parser.add_argument("--num_views", type=int, default=5, help="Number of test views to render")
    parser.add_argument("--quiet", action="store_true", help="Disable progress output")
    
    # Add options for handling the test view issue
    parser.add_argument("--use_train_for_test", action="store_true", help="Use training views for testing")
    
    args = parser.parse_args()
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # Extract parameters
    dataset = model.extract(args)
    opt = op.extract(args)
    pipe = pipeline.extract(args)
    
    # Load model
    gaussians, scene = load_model(args.checkpoint, dataset, opt)
    
    # Check if we need to use training views for testing
    test_views = scene.getTestCameras()
    train_views = scene.getTrainCameras()
    
    if len(test_views) == 0:
        print("Warning: No test views found in the dataset!")
        if args.use_train_for_test:
            print("Using training views for testing as requested")
            # Take a subset of train views to use as test views
            num_test = min(20, len(train_views) // 5)  # Use up to 20 views or 20% of train views
            # Choose evenly spaced views for diversity
            test_indices = np.linspace(0, len(train_views)-1, num_test, dtype=int)
            test_views = [train_views[i] for i in test_indices]
            # Assign these views as test cameras
            scene._test_cameras = test_views
            print(f"Created {len(test_views)} test views from training views")
    
    # Define background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Precompute H_per_gaussian using training and testing views
    print("Precomputing Hessian values per Gaussian...")
    H_per_gaussian = precompute_H_per_gaussian(gaussians, scene, pipe, background)
    
    # Render uncertainty from test views
    print("Rendering uncertainty visualizations...")
    render_uncertainty_from_test_views(scene, gaussians, pipe, background, args.output_dir, H_per_gaussian, args.num_views)
    print(f"Uncertainty renders and plots saved to {args.output_dir}")