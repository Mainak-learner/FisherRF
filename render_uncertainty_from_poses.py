import torch
import os
import numpy as np
from scene import Scene, GaussianModel
from gaussian_renderer import render, forward_k_times, modified_render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
from scene.cameras import Camera
import torchvision
import json
from einops import reduce
import matplotlib.pyplot as plt

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

def render_uncertainty_from_poses(poses, gaussians, pipe, background, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(15, 5 * len(poses)))  # Adjusted for 3 columns

    for idx, pose in enumerate(poses):
        # Convert pose to Camera object with image=None
        viewpoint = Camera(
            colmap_id=idx,
            R=np.array(pose["R"]),
            T=np.array(pose["T"]),
            FoVx=pose["FoVx"],
            FoVy=pose["FoVy"],
            image=None,  # Explicitly None for novel poses
            gt_alpha_mask=None,
            image_name=f"pose_{idx:03d}",
            uid=idx,
            data_device="cuda"
        )

        # Variational Uncertainty
        variational_pkg = forward_k_times(viewpoint, gaussians, pipe, background, k=gaussians.n_models)
        var_rgb = variational_pkg["comp_rgb"]
        var_uncertainty = variational_pkg["comp_std"]  # Standard deviation

        # FisherRF Uncertainty
        render_pkg = modified_render(viewpoint, gaussians, pipe, background)
        fisher_rgb = render_pkg["render"]
        depth = render_pkg["depth"]
        xyz = gaussians._xyz
        to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)], dim=-1)
        pts3d_homo = to_homo(xyz)
        pts3d_cam = pts3d_homo @ viewpoint.world_view_transform
        gaussian_depths = pts3d_cam[:, 2, None].clamp(min=0)

        # Compute FisherRF uncertainty with depth normalization
        hessian_color = torch.ones_like(xyz)  # Placeholder; adjust if needed
        cur_hessian_color = hessian_color * gaussian_depths
        fisher_render_pkg = render(viewpoint, gaussians, pipe, background, override_color=cur_hessian_color)
        fisher_uncertainty = reduce(fisher_render_pkg["render"], "c h w -> h w", "mean")
        pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
        fisher_uncertainty = torch.log(fisher_uncertainty / pixel_gaussian_counter.clamp(min=1e-6))

        # Normalize uncertainties to 0-1
        var_min, var_max = var_uncertainty.min(), var_uncertainty.max()
        var_uncertainty_norm = (var_uncertainty - var_min) / (var_max - var_min + 1e-6)

        fisher_min, fisher_max = fisher_uncertainty.min(), fisher_uncertainty.max()
        fisher_uncertainty_norm = (fisher_uncertainty - fisher_min) / (fisher_max - fisher_min + 1e-6)

        # Convert tensors to numpy for plotting
        render_img_np = var_rgb.cpu().numpy().transpose(1, 2, 0)
        fisher_unc_np = fisher_uncertainty_norm.cpu().numpy()
        var_unc_np = var_uncertainty_norm.cpu().numpy()

        # Plotting (3 subplots per row)
        plt.subplot(len(poses), 3, idx * 3 + 1)
        plt.imshow(render_img_np)
        plt.title(f"Render - Pose {idx}")
        plt.axis('off')

        plt.subplot(len(poses), 3, idx * 3 + 2)
        plt.imshow(fisher_unc_np, cmap='viridis')
        plt.title(f"FisherRF Unc - Pose {idx}")
        plt.axis('off')

        plt.subplot(len(poses), 3, idx * 3 + 3)
        plt.imshow(var_unc_np, cmap='viridis')
        plt.title(f"Var Unc - Pose {idx}")
        plt.axis('off')

        # Save images
        side_by_side_rgb = torch.cat([var_rgb, fisher_rgb], dim=2)
        side_by_side_unc = torch.cat([var_uncertainty_norm, fisher_uncertainty_norm], dim=1)
        torchvision.utils.save_image(side_by_side_rgb, os.path.join(output_dir, f"rgb_{idx:03d}.png"))
        torchvision.utils.save_image(side_by_side_unc, os.path.join(output_dir, f"uncertainty_{idx:03d}.png"))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "uncertainty_plots.png"))
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser(description="Render uncertainty from novel poses")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--poses_file", type=str, required=True, help="Path to JSON file with novel poses")
    parser.add_argument("--output_dir", type=str, default="./uncertainty_renders", help="Output directory for renders")

    args = parser.parse_args()

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    gaussians, scene = load_model(args.checkpoint, dataset, opt)

    with open(args.poses_file, "r") as f:
        poses = json.load(f)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_uncertainty_from_poses(poses, gaussians, pipe, background, args.output_dir)
    print(f"Uncertainty renders and plots saved to {args.output_dir}")