#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, forward_k_times
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from utils.camera_utils import rand_rotation_matrix
from scene.cameras import Camera
from gaussian_renderer import modified_render
from einops import reduce, repeat, rearrange
from utils.load_custom_poses import load_cameras_from_pose_file
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from active.schema import schema_dict, override_test_idxs_dict, override_train_idxs_dict
from scene import Scene
import json
from vis.launch_viewer import launch_viewer_from_json
import random

def capture(self):
    return (
        self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        self.xyz_gradient_accum,
        self.denom,
        # self.optimizer.state_dict(),
        # self.spatial_lr_scale,
    )

@torch.no_grad()
def render_uncertainty(view, gaussians, pipeline, background, hessian_color):
    render_pkg = modified_render(view, gaussians, pipeline, background)
    pred_img = render_pkg["render"]
    # pred_img.backward(gradient=torch.ones_like(pred_img))
    pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]

    render_pkg = modified_render(view, gaussians, pipeline, background, override_color=hessian_color)

    uncertanity_map = reduce(render_pkg["render"], "c h w -> h w", "mean")

    return pred_img, uncertanity_map, pixel_gaussian_counter, render_pkg["depth"]

def render_set(model_path, name, iteration, train_views, test_views, gaussians, pipeline, background, perturb_scale=1., camera_extent=None, args=None):
    render_path = os.path.join(model_path, "renders")
    eval_path = os.path.join(model_path, "eval")

    makedirs(render_path, exist_ok=True)
    makedirs(eval_path, exist_ok=True)

    params = capture(gaussians)[1:7]
    name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
    xyz = params[0]
    # filter_out_idx = [name2idx[k] for k in ["rotation", "rgb", "sh"]]
    filter_out_idx = [name2idx[k] for k in ["rotation", "scale", "xyz", "opacity"]]
    params = [p.requires_grad_(True) for i, p in enumerate(params) if i not in filter_out_idx]
    optim = torch.optim.SGD(params, 0.)
    gaussians.optimizer = optim
    device = params[0].device
    # H_train = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype)
    H_per_gaussian = torch.zeros(params[0].shape[0], device=params[0].device, dtype=params[0].dtype)

    if not args.depth_only:
        # TODO: We can also use all the views, here the train views are just a subset of training cameras
        for idx, view in enumerate(tqdm(itertools.chain(train_views, test_views), desc="Rendering progress")):

            # rendering = render(view, gaussians, pipeline, background)["render"]

            render_pkg = modified_render(view, gaussians, pipeline, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))
            pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
            # render_pkg = modified_render(view, gaussians, pipeline, background, override_color=torch.ones_like(params[1]))
            H_per_gaussian += sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in params])
            # render_pkg = modified_render(view, gaussians, pipeline, background, override_color=H_per_gaussian.detach())
            optim.zero_grad(set_to_none = True) 

            split = "train" if idx < len(train_views) else "test"

            torchvision.utils.save_image(pred_img.detach(), os.path.join(render_path, f"{split}_{view.image_name}.png"))
    else:
        H_per_gaussian += 1

    hessian_color = repeat(H_per_gaussian.detach(), "n -> n c", c=3)
    
    with torch.no_grad():
        for idx, view in enumerate(tqdm(test_views, desc="Rendering on test set")):
            
            to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype, device=x.device)], dim=-1)
            pts3d_homo = to_homo(xyz)
            pts3d_cam = pts3d_homo @ view.world_view_transform
            gaussian_depths = pts3d_cam[:, 2, None]

            cur_hessian_color = hessian_color * gaussian_depths.clamp(min=0)

            pred_img, uncertanity_map, pixel_gaussian_counter, depth = render_uncertainty(view, gaussians, pipeline, background, cur_hessian_color)

            # sns.heatmap(torch.log(uncertanity_map / pixel_gaussian_counter).clamp(min=0).detach().cpu(), square=True)
            # plt.savefig(f"./uncern_all.jpg")
            # torchvision.utils.save_image(pred_img.detach(), os.path.join(render_path, f"{split}_{idx:05d}.png"))
            if args.depth_only:
                sns.heatmap(depth.detach().cpu(), square=True)
                plt.savefig(os.path.join(eval_path, f"depth_viz_{view.image_name}.jpg"))
            else:
                sns.heatmap(torch.log(uncertanity_map / pixel_gaussian_counter).detach().cpu(), square=True)
                plt.savefig(os.path.join(eval_path, f"heatmap_{view.image_name}.jpg"))
            plt.clf()

            np.savez(os.path.join(eval_path, f"uncertainty_{idx:03d}_{view.image_name}.npz"), 
                     uncertanity_map=uncertanity_map.cpu(), pixel_gaussian_counter=pixel_gaussian_counter.cpu(),
                     depth=depth.cpu(),
                     )

def render_set_current(model_path, name, iteration, train_views, test_views, gaussians, pipeline, background, perturb_scale=1., camera_extent=None, args=None):
    eval_path = os.path.join(model_path, "eval")

    makedirs(eval_path, exist_ok=True)

    params = capture(gaussians)[1:7]
    name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
    filter_out_idx = [name2idx[k] for k in ["rotation"]]
    params = [p.requires_grad_(True) for i, p in enumerate(params) if i not in filter_out_idx]
    optim = torch.optim.SGD(params, 0.)
    gaussians.optimizer = optim
    device = params[0].device

    for idx, view in enumerate(tqdm(test_views, desc="Rendering on test set")):

        render_pkg = modified_render(view, gaussians, pipeline, background)
        pred_img = render_pkg["render"]
        pred_img.backward(gradient=torch.ones_like(pred_img))
        pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]
        H_per_gaussian = sum(reduce(p.grad.detach(), "n ... -> n", "sum") for p in params)

        with torch.no_grad():
            hessian_color = repeat(H_per_gaussian.detach(), "n -> n c", c=3)

            # compute depth of gaussian in current view
            to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype, device=x.device)], dim=-1)
            pts3d_homo = to_homo(params[0])
            pts3d_cam = pts3d_homo @ view.world_view_transform
            gaussian_depths = pts3d_cam[:, 2, None]

            hessian_color = hessian_color * gaussian_depths

            render_pkg = modified_render(view, gaussians, pipeline, background, override_color=hessian_color)

            uncertanity_map = reduce(render_pkg["render"], "c h w -> h w", "mean")
            depth = render_pkg["depth"]

            # sns.heatmap(torch.log(uncertanity_map / pixel_gaussian_counter).clamp(min=0).detach().cpu(), square=True)
            # plt.savefig(f"./uncern.jpg")
            # plt.savefig(f"./uncern_all.jpg")
            plt.clf()

            torchvision.utils.save_image(pred_img.detach(), os.path.join(eval_path, f"render_{view.image_name}.png"))
            sns.heatmap(torch.log(uncertanity_map / pixel_gaussian_counter).clamp(min=0).detach().cpu(), square=True)
            plt.savefig(os.path.join(eval_path, f"heatmap_{view.image_name}.jpg"))
            plt.clf()

            np.savez(os.path.join(eval_path, f"uncertainty_{idx:03d}_{view.image_name}.npz"), 
                     uncertanity_map=uncertanity_map.cpu(), pixel_gaussian_counter=pixel_gaussian_counter.cpu(),
                     depth=depth.cpu(),
                     )

            optim.zero_grad(set_to_none = True) 

# New function that combines FisherRF and variational uncertainty computation
def render_combined_uncertainty(model_path, name, iteration, train_views, test_views, gaussians, pipeline, background, num_views=5, perturb_scale=1., camera_extent=None, args=None):
    """Render and compare both FisherRF and variational uncertainty visualization."""
    render_path = os.path.join(model_path, "renders")
    eval_path = os.path.join(model_path, "eval")
    combined_path = os.path.join(model_path, "new_combined_uncertainty")

    makedirs(render_path, exist_ok=True)
    makedirs(eval_path, exist_ok=True)
    makedirs(combined_path, exist_ok=True)

    # If no test views, use a subset of train views instead
    if len(test_views) == 0:
        print("No test views found. Using a subset of training views...")
        # if len(train_views) <= num_views:
        #     selected_views = train_views
        # else:
        #     # Randomly select views
        #     selected_indices = random.sample(range(len(train_views)), num_views)
        #     selected_views = [train_views[i] for i in selected_indices]
        selected_views = train_views        
    else:
        selected_views = test_views        
        # Use actual test views
        # if len(test_views) <= num_views:
        #     selected_views = test_views
        # else:
        #     # Randomly select views
        #     selected_indices = random.sample(range(len(test_views)), num_views)
        #     selected_views = [test_views[i] for i in selected_indices]
    
    print(f"Selected {len(selected_views)} views for uncertainty visualization")

    # Compute FisherRF uncertainty
    # Extract parameters
    params = capture(gaussians)[1:7]
    name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
    xyz = params[0]
    # Exclude rotation, scale, xyz, opacity from gradient computation
    filter_out_idx = [name2idx[k] for k in ["rotation", "scale", "xyz", "opacity"]]
    params = [p.requires_grad_(True) for i, p in enumerate(params) if i not in filter_out_idx]
    optim = torch.optim.SGD(params, 0.)
    gaussians.optimizer = optim
    device = params[0].device
    
    # Initialize H_per_gaussian tensor
    H_per_gaussian = torch.zeros(params[0].shape[0], device=params[0].device, dtype=params[0].dtype)

    # Compute H_per_gaussian using all views (train + test)
    if not args.depth_only:
        print("Computing FisherRF uncertainty with all views...")
        for idx, view in enumerate(tqdm(train_views, desc="Computing FisherRF uncertainty")):
            render_pkg = modified_render(view, gaussians, pipeline, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))
            H_per_gaussian += sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in params])
            optim.zero_grad(set_to_none=True)
    else:
        H_per_gaussian += 1

    # Prepare color from Hessian
    hessian_color = repeat(H_per_gaussian.detach(), "n -> n c", c=3)
    
    fisher_unc_norms = []
    # Render uncertainty visualizations for selected views
    with torch.no_grad():
        for idx, view in enumerate(tqdm(selected_views, desc="Rendering combined uncertainty visualization")):
            # Compute depth for FisherRF
            to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)], dim=-1)
            pts3d_homo = to_homo(xyz)
            pts3d_cam = pts3d_homo @ view.world_view_transform
            gaussian_depths = pts3d_cam[:, 2, None]
            
            # Scale hessian color by depth
            cur_hessian_color = hessian_color * gaussian_depths.clamp(min=0)
            
            # Render FisherRF uncertainty
            pred_img, fisher_uncertainty_map, pixel_gaussian_counter, depth = render_uncertainty(
                view, gaussians, pipeline, background, cur_hessian_color
            )
            
            # Normalize FisherRF uncertainty for visualization
            fisher_unc_norm = torch.log(fisher_uncertainty_map / pixel_gaussian_counter.clamp(min=1e-6))

            min_fisher_val = fisher_unc_norm.min()
            max_fisher_val = fisher_unc_norm.max()
            if max_fisher_val > min_fisher_val:
                fisher_unc_norm = (fisher_unc_norm - min_fisher_val) / (max_fisher_val - min_fisher_val)
            else:
                fisher_unc_norm = torch.zeros_like(fisher_unc_norm)
            
            fisher_unc_norms.append(fisher_unc_norm)
            # Render variational uncertainty if available
            if hasattr(gaussians, 'n_models') and gaussians.n_models > 1:
                # Variational Uncertainty
                print(f"Computing variational uncertainty for view {idx}...")
                variational_pkg = forward_k_times(view, gaussians, pipeline, background, k=gaussians.n_models)
                var_rgb = variational_pkg["comp_rgb"].detach()
                var_uncertainty = 10 * variational_pkg["comp_std"].detach()
                # Convert to single-channel uncertainty
                var_uncertainty_map = var_uncertainty.mean(dim=0)  # Shape: (height, width)
                
                min_var_val = var_uncertainty_map.min()
                max_var_val = var_uncertainty_map.max()
                if max_var_val > min_var_val:
                    var_uncertainty_map = (var_uncertainty_map - min_var_val) / (max_var_val - min_var_val)
                else:
                    var_uncertainty_map = torch.zeros_like(var_uncertainty_map)

                if hasattr(args, "pose_json"):
                    prefix = "pose_"
                elif hasattr(args, "generate_custom_from_test_train"):
                    prefix = "generated_pose_"
                else:
                    prefix = "combined_uncertainty_"
                # Create visualization
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                
                # Show RGB render
                axs[0, 0].imshow(pred_img.permute(1, 2, 0).cpu().numpy())
                axs[0, 0].set_title(f"Render - View {view.image_name}")
                axs[0, 0].axis('off')
                
                # Show depth
                depth_img = axs[0, 1].imshow(depth.cpu().numpy(), cmap='magma')
                axs[0, 1].set_title("Depth")
                axs[0, 1].axis('off')
                plt.colorbar(depth_img, ax=axs[0, 1], fraction=0.046, pad=0.04)
                
                # Show FisherRF uncertainty
                fisher_viz_img = axs[1, 0].imshow(fisher_unc_norm.clamp(min=0).cpu().numpy(), cmap='magma')
                axs[1, 0].set_title("FisherRF Uncertainty")
                axs[1, 0].axis('off')
                plt.colorbar(fisher_viz_img, ax=axs[1, 0], fraction=0.046, pad=0.04)
                
                # Show variational uncertainty
                var_viz_img = axs[1, 1].imshow(var_uncertainty_map.cpu().numpy(), cmap='magma')  
                axs[1, 1].set_title("Variational Uncertainty")
                axs[1, 1].axis('off')
                plt.colorbar(var_viz_img, ax=axs[1, 1], fraction=0.046, pad=0.04)

                
                plt.tight_layout()
                plt.savefig(os.path.join(combined_path, f"{prefix}{view.image_name}.png"))
                plt.close()
                
                # Save images separately
                torchvision.utils.save_image(pred_img.detach(), os.path.join(combined_path, f"render_{view.image_name}.png"))
                
                # Save raw data for further analysis
                np.savez(
                    os.path.join(combined_path, f"data_{view.image_name}.npz"),
                    fisher_uncertainty=fisher_unc_norm.cpu().numpy(),
                    var_uncertainty=var_uncertainty_map.cpu().numpy(),
                    pixel_gaussian_counter=pixel_gaussian_counter.cpu().numpy(),
                    depth=depth.cpu().numpy()
                )
            else:
                # Only FisherRF is available (no variational)
                print("No variational model detected, rendering FisherRF uncertainty only")
                
                # Create visualization
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                
                # Show RGB render
                axs[0].imshow(pred_img.permute(1, 2, 0).cpu().numpy())
                axs[0].set_title(f"Render - View {view.image_name}")
                axs[0].axis('off')
                
                # Show depth
                axs[1].imshow(depth.cpu().numpy(), cmap='viridis')
                axs[1].set_title("Depth")
                axs[1].axis('off')
                
                # Show FisherRF uncertainty
                fisher_viz = fisher_unc_norm.clamp(min=0).cpu().numpy()
                axs[2].imshow(fisher_viz, cmap='viridis')
                axs[2].set_title("FisherRF Uncertainty")
                axs[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(combined_path, f"fisher_uncertainty_{view.image_name}.png"))
                plt.close()
                
                # Save images separately
                torchvision.utils.save_image(pred_img.detach(), os.path.join(combined_path, f"render_{view.image_name}.png"))
                
                # Save raw data for further analysis
                np.savez(
                    os.path.join(combined_path, f"data_{view.image_name}.npz"),
                    fisher_uncertainty=fisher_unc_norm.cpu().numpy(),
                    pixel_gaussian_counter=pixel_gaussian_counter.cpu().numpy(),
                    depth=depth.cpu().numpy()
                )
    
    return fisher_unc_norms, selected_views

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, args):
    # Initialize Gaussian model - if args.use_variational is true, create variational model
    is_variational = args.use_variational if hasattr(args, 'use_variational') else False
    gaussians = GaussianModel(dataset.sh_degree, is_variational=is_variational)
    print(f"Created {'variational' if is_variational else 'standard'} Gaussian model")

    # Create scene with appropriate training and test views
    if hasattr(args, 'override_idxs') and args.override_idxs is not None:
        override_train_idxs = list(range(10_000))  # Use all frames for training
        override_test_idxs = override_test_idxs_dict.get(args.override_idxs, [])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, 
                     override_train_idxs=override_train_idxs, override_test_idxs=override_test_idxs)
    else:
        # Standard scene creation
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    # Print camera counts
    train_views = scene.getTrainCameras()

    if hasattr(args, "pose_json") and args.pose_json:
        print("Using custom poses from:", args.pose_json)
        fallback_res = (train_views[0].image_height, train_views[0].image_width)
        custom_views = load_cameras_from_pose_file(args.pose_json, device="cuda", resolution=fallback_res)
        test_views = custom_views

    elif hasattr(args, "generate_custom_from_test_train") and args.generate_custom_from_test_train:
        print("Generating perturbed custom poses from:", args.generate_custom_from_test_train)

        # Generate perturbed poses and write to a temp file
        import tempfile
        from scene.gen_custom_poses import perturb_and_generate_poses

        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False, encoding="utf-8") as tmp:
            perturb_and_generate_poses(
                scene=scene,
                gaussians = gaussians,
                output_json_path=tmp.name,
                n_poses=args.num_generated_poses,
                radius_perturb=args.tangent_perturb
            )
            tmp_path = tmp.name

        fallback_res = (train_views[0].image_height, train_views[0].image_width)
        custom_views = load_cameras_from_pose_file(tmp_path, device="cuda", resolution=fallback_res)
        test_views = custom_views

    else:
        test_views = scene.getTestCameras()

    
    print(f"Loaded scene with {len(train_views)} training views and {len(test_views)} test views")

    # Set background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Render with appropriate method
    if args.combined:
        # New mode - render both FisherRF and variational uncertainty
        fisher_unc_norms, selected_views = render_combined_uncertainty(
                dataset.model_path, "train", scene.loaded_iter, 
                train_views, test_views, 
                gaussians, pipeline, background, 
                num_views=args.num_views if hasattr(args, 'num_views') else 5,
                camera_extent=scene.cameras_extent, args=args
            )
        
        pose_entries = []
        for view, unc in zip(selected_views, fisher_unc_norms):  # assuming test_uncertainties already computed
            pos = view.camera_center.cpu().numpy().tolist()
            direction = -(view.R.T @ torch.tensor([0., 0., 1.], device=view.R.device))  # -Z axis
            dir = direction.cpu().numpy().tolist()
            unc_mean = unc.mean().cpu().numpy()
            pose_entries.append({
                "position": pos,
                "direction": dir,
                "uncertainty": float(unc_mean),
                "FoVx": float(view.FoVx),
                "FoVy": float(view.FoVy)
            })

        json_path = "vis_output/nbv_custom_poses.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(pose_entries, f, indent=2)
        launch_viewer_from_json(json_path, ngrok_token="YOUR_REAL_NGROK_TOKEN")
        np.save("vis_output/object_xyz.npy", scene.gaussians._xyz.detach().cpu().numpy())
    elif args.current:
        # Original "current" mode - compute uncertainty for each view independently
        render_set_current(
            dataset.model_path, "train", scene.loaded_iter, 
            train_views, test_views, 
            gaussians, pipeline, background, 
            camera_extent=scene.cameras_extent, args=args
        )
    else:
        # Original mode - precompute uncertainty using all views, then visualize
        render_set(
            dataset.model_path, "train", scene.loaded_iter, 
            train_views, test_views, 
            gaussians, pipeline, background, 
            camera_extent=scene.cameras_extent, args=args
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    # Original arguments
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--perturb_scale", default=1., type=float)
    parser.add_argument("--inflate_factor", default=5, type=int)
    parser.add_argument("--override_idxs", type=str, help="special test idxs on uncertainty evaluation")
    parser.add_argument("--depth_only", action="store_true", help="render depth only")
    parser.add_argument("--current", action="store_true", help="render uncertainty from current view")
    parser.add_argument("--pose_json", type=str, default=None, help="Path to a JSON file of custom camera poses")
    parser.add_argument("--generate_custom_from_test_train", type=str, default=None,
                    help="Path to training and test transform JSON from which to auto-generate spherical poses")
    parser.add_argument("--num_generated_poses", type=int, default=10, help="Number of generated custom poses")
    parser.add_argument("--tangent_perturb", type=float, default=0.05, help="Degree of perturbation along tangent")

    
    # New arguments for variational mode
    parser.add_argument("--use_variational", action="store_true", help="Use variational Gaussian model")
    parser.add_argument("--combined", action="store_true", help="Render both FisherRF and variational uncertainty")
    parser.add_argument("--num_views", type=int, default=5, help="Number of views to render for combined mode")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args)