# active_train_sector.py
import os
import torch
import wandb
import numpy as np
import uuid
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips_func
from active.lpips_selector import LPIPSNBVSelector
from active.encoders import ImageEncoder, PoseToImageEncoder
from scene.sector_pose_gen import generate_circular_hemisphere_poses, divide_hemisphere_poses, sample_uniform_sphere_views_disjoint
from utils.camera_utils import look_at, look_at_torch
from utils.graphics_utils import uv2car_torch
from scene.cameras import DummyCamera
from torchvision.utils import save_image
import base64
import json
from PIL import Image
from active.mc_dkl_selector import MCDKLNBVSelector
from active.gp_predictor import GPFisherNBVSelector, VDGPFisherNBVSelector
from gaussian_renderer import render, network_gui, modified_render
import torchvision.transforms.functional as TF
from arguments import ModelParams, PipelineParams, OptimizationParams
from active.train_phi_pose_to_feat import train_phi_sector
import torch.backends.cudnn as cudnn
import random
import torch.nn.functional as F
import pandas as pd
from copy import deepcopy

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def get_robust_init_pose(proposal_uvs, uncertainties, u_bounds, v_bounds, sample_radius, strategy="topk-centroid", k=5, eval_selected_cams=None):
    if strategy == "weighted":
        w = uncertainties / (uncertainties.sum() + 1e-8)
        w = w.detach().cpu().numpy()
        return tuple(np.sum(w[:, None] * np.array(proposal_uvs), axis=0))
    elif strategy == "topk-centroid":
        topk_idx = torch.topk(uncertainties, k=k).indices
        topk_uvs = np.array([proposal_uvs[i] for i in topk_idx])
        return tuple(np.mean(topk_uvs, axis=0))
    elif strategy == "diverse-fisher":
        scores = []
        for i, uv in enumerate(proposal_uvs):
            uv_tensor = torch.tensor(uv, dtype=torch.float32, device="cuda")
            dist = min([
                    np.linalg.norm(
                        ((uv2car_torch(uv_tensor[0].unsqueeze(0), uv_tensor[1].unsqueeze(0)) * sample_radius) - cam.camera_center).detach().cpu().numpy()
                    )
                    for cam in eval_selected_cams
                    ])
            scores.append(uncertainties[i].item() * dist)
        return proposal_uvs[np.argmax(scores)]
    else:  # default to midpoint
        return ((u_bounds[0] + u_bounds[1]) / 2, (v_bounds[0] + v_bounds[1]) / 2)

def init_metric_logger(output_dir, filename="metrics_log.csv"):
    log_path = os.path.join(output_dir, filename)
    if not os.path.exists(log_path):
        df = pd.DataFrame(columns=["Iteration", "PSNR", "SSIM", "LPIPS"])
        df.to_csv(log_path, index=False)
    return log_path

def log_metrics(log_path, iteration, psnr_value, ssim_value, lpips_value):
    df = pd.read_csv(log_path)
    df = pd.concat([df, pd.DataFrame([{
        "Iteration": iteration,
        "PSNR": round(psnr_value, 2),
        "SSIM": round(ssim_value, 4),
        "LPIPS": round(lpips_value, 4)
    }])], ignore_index=True)
    df.to_csv(log_path, index=False)

def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # important for deterministic matmul
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # Throws if you use nondeterministic ops


def render_fn(cam_center, object_center, pipe, gaussians, background, reference_camera, debug=False):
    R, T = look_at(cam_center.detach(), object_center.detach(), debug)
    dummy_cam = DummyCamera(R, T, reference_camera)
    return render(dummy_cam, gaussians, pipe, background)["render"]

def render_with_oracle(cam_center, object_center, pipe, oracle_gaussians, background, reference_camera):
    R, T = look_at(cam_center.detach(), object_center.detach())
    dummy_cam = DummyCamera(R, T, reference_camera)
    return render(dummy_cam, oracle_gaussians, pipe, background)["render"]

def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[:10])
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))
    return None

def compute_fisher_hessian(gaussians, cameras, pipe, background, filter_out_idx, reg_lambda):
    params = gaussians.capture()[1:7]
    params = [p for i, p in enumerate(params) if i not in filter_out_idx]
    H_train = torch.zeros(sum(p.numel() for p in params), device=params[0].device)

    for cam in tqdm(cameras, desc="Caching diagonal Hessian on training views"):
        render_pkg = modified_render(cam, gaussians, pipe, background)
        pred_img = render_pkg["render"]
        pred_img.backward(gradient=torch.ones_like(pred_img))

        cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])
        H_train += cur_H

        gaussians.optimizer.zero_grad(set_to_none=True)

    return torch.reciprocal(H_train + reg_lambda)

def training(dataset, opt, pipe, test_iterations, save_iterations, args):
    prepare_output_and_logger(dataset)
    set_global_seed(args.seed)
    log_path = init_metric_logger(args.model_path)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    oracle_gaussians = GaussianModel(dataset.sh_degree)
    oracle_gaussians.load_ply(os.path.join(args.oracle_model_path, "point_cloud/iteration_30000/point_cloud.ply"))

    object_center = oracle_gaussians.get_xyz.mean(dim=0).detach()
    reference_camera = scene.getAllCameras()[0]
    sample_radius = torch.norm(reference_camera.camera_center).item()

    all_centers, all_uvs, pose_per_circle = generate_circular_hemisphere_poses(torch.tensor([0, 0, 0], device=object_center.device), num_circles=args.num_circles, min_poses=args.min_poses, radius=sample_radius)
    circle_indices, middle_circle_indices, sector_map = divide_hemisphere_poses(all_centers, object_center.cpu().numpy(), pose_per_circle, num_circles=args.num_circles)
    
    os.makedirs("oracle_gt_visualization", exist_ok=True)
    np.save("oracle_gt_visualization/object_center.npy", object_center.cpu().numpy())
    np.save("oracle_gt_visualization/object_points.npy", oracle_gaussians.get_xyz.detach().cpu().numpy())
    np.save("oracle_gt_visualization/object_colors.npy", oracle_gaussians._features_dc.detach().cpu().numpy())
    # for idx, cam_center in enumerate(tqdm(all_centers, desc="Rendering Oracle GT Images")):
    #     cam_center = all_centers[idx]
    #     gt_img = render_with_oracle(cam_center, object_center, pipe, oracle_gaussians, torch.tensor([1.0, 1.0, 1.0], device="cuda"), reference_camera)
    #     img_path = f"oracle_gt_visualization/proposal_pose_{idx}.png"
    #     TF.to_pil_image(gt_img.cpu()).save(img_path) 
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    
    # Sample test views with explicit disjointness from proposals
    uniform_centers, directions = sample_uniform_sphere_views_disjoint(
        num_views=400,
        radius=sample_radius,
        object_center=object_center,
        exclude_centers=all_centers,   # make tests disjoint from proposals
        angle_deg=1.0,                 # tune as you like (1–3 degrees is common)
        oversample_factor=2
    )

    uniform_test_cameras = []
    for center, direction in zip(uniform_centers, directions):
        cam_params = look_at(center, object_center, reference_camera)
        cam = DummyCamera(*cam_params, reference_camera)
        uniform_test_cameras.append(cam)

    oracle_renders = []
    for cam_center in tqdm(uniform_centers):
        with torch.no_grad():
            rendered = render_with_oracle(cam_center, object_center, pipe, oracle_gaussians, torch.tensor([1.0, 1.0, 1.0], device="cuda"), reference_camera)
        oracle_renders.append(rendered.clamp(0, 1))
    middle_ids = np.random.choice(middle_circle_indices, size=6, replace=False)
    selected_cams = []

    selected_middle_centers = []
    oracle_image_paths = []
    selected_cams = []

    # Start with the first index
    selected_indices = [middle_ids[0]]
    selected_middle_centers.append(all_centers[middle_ids[0]].cpu().numpy().tolist())

    while len(selected_indices) < len(middle_ids):
        selected = torch.stack([all_centers[idx] for idx in selected_indices])  # (S, 3)
        remaining = list(set(middle_ids) - set(selected_indices))

        # Compute distance of each remaining point to the closest selected point
        dists = []
        for idx in remaining:
            candidate = all_centers[idx].unsqueeze(0)  # (1, 3)
            dist = torch.cdist(candidate, selected).min().item()
            dists.append((dist, idx))
        
        # Pick the one with maximum min-distance
        _, next_idx = max(dists, key=lambda x: x[0])
        selected_indices.append(next_idx)
        selected_middle_centers.append(all_centers[next_idx].cpu().numpy().tolist())

    # Now render and save images for each selected index
    for i, idx in enumerate(selected_indices):
        cam_center = all_centers[idx]
        gt_img = render_with_oracle(cam_center, object_center, pipe, oracle_gaussians, torch.tensor([1.0, 1.0, 1.0], device="cuda"), reference_camera)
        img_path = f"oracle_gt_visualization/pose_{i}.png"
        TF.to_pil_image(gt_img.clamp(0, 1).cpu()).save(img_path)

        oracle_image_paths.append(f"pose_{i}.png")
        dummy_camera = DummyCamera(*look_at(cam_center.detach(), object_center.detach()), reference_camera, image=gt_img.detach())
        print(f"cam_center:{cam_center}, dummy_cam_center:{dummy_camera.camera_center}") 
        selected_cams.append(dummy_camera)


    viewpoint_stack=None 
    for iteration in tqdm(range(1, args.initial_train + 1), desc="Initial Training on Middle Circle"):
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = selected_cams.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, args.min_opacity, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < args.initial_train:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
    
    lpips_metric = lpips_func("cuda", net_type='vgg')
    psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
    test_cams = scene.getAllCameras(1.0)

    for i, cam in enumerate(uniform_test_cameras):
        rendered = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
        psnr_total += psnr(rendered, oracle_renders[i]).mean().item()
        ssim_total += ssim(rendered, oracle_renders[i]).mean().item()
        lpips_total += lpips_metric(rendered, oracle_renders[i]).mean().item()

    num_eval = len(uniform_test_cameras)
    avg_psnr = psnr_total / num_eval
    avg_ssim = ssim_total / num_eval
    avg_lpips = lpips_total / num_eval

    print(f"[Iter 0] PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
    log_metrics(log_path, 0, avg_psnr, avg_ssim, avg_lpips)
    
    eval_selected_cams = selected_cams.copy()  # for evaluation + GP selector
    used_uvs_global = set()
    if args.vdgp:
        pose_feat_dim = 3  # or 6 if using orientation
        image_feat_dim = 128  # same as Φ output dim   
    else:
        selector = GPFisherNBVSelector(args, device="cuda")

    gaussians_dict = {
        "rbf": deepcopy(gaussians),
        "matern": deepcopy(gaussians),
        "rq": deepcopy(gaussians),
        "linear": deepcopy(gaussians),
        "periodic": deepcopy(gaussians),
        "spectral": deepcopy(gaussians),
    }

    selected_cams_dict = {
        "rbf": deepcopy(selected_cams),
        "matern": deepcopy(selected_cams),
        "rq": deepcopy(selected_cams),
        "linear": deepcopy(selected_cams),
        "periodic": deepcopy(selected_cams),
        "spectral": deepcopy(selected_cams),
    }

    eval_selected_cams_dict = {
        "rbf": deepcopy(selected_cams),
        "matern": deepcopy(selected_cams),
        "rq": deepcopy(selected_cams),
        "linear": deepcopy(selected_cams),
        "periodic": deepcopy(selected_cams),
        "spectral": deepcopy(selected_cams),
    }

    # Kernel ablation loop
    for kernel in ["rbf", "matern", "rq", "linear", "periodic", "spectral"]:
        print(f"\n=== Starting sector-based training with kernel: {kernel} ===")

        gaussians = gaussians_dict[kernel]
        selected_cams = selected_cams_dict[kernel]
        eval_selected_cams = eval_selected_cams_dict[kernel]


        used_uvs_global = set()

        for train_iter in range(1, args.max_nbv_iterations + 1):

            print(f"=== Iteration {train_iter}: selecting NBVs from sectors ===")

            sector_selections = []
            sector_init_poses = []

            if args.vdgp or args.deepkgp:
                I_train_diag = compute_fisher_hessian(
                gaussians, eval_selected_cams, pipe, background,
                selector.filter_out_idx, selector.reg_lambda
                )
            for sector_id, sector_indices in sector_map.items():
                sector_dir = f"oracle_gt_visualization_sectorwise/sector_{sector_id}"
                os.makedirs(sector_dir, exist_ok=True)

                if len(sector_indices) == 0:
                    continue

                proposal_uvs = [all_uvs[idx] for idx in sector_indices]
                proposal_centers = all_centers[sector_indices]

                u_vals = [uv[0] for uv in proposal_uvs]
                v_vals = [uv[1] for uv in proposal_uvs]
                u_bounds = (min(u_vals), max(u_vals))
                v_bounds = (min(v_vals), max(v_vals))

                # Grid resolution per sector (e.g., 20x20 = 400 samples)
                grid_res = 20  

                u_vals = np.linspace(u_bounds[0], u_bounds[1], grid_res)
                v_vals = np.linspace(v_bounds[0], v_bounds[1], grid_res)

                dense_uvs = np.array([(u, v) for u in u_vals for v in v_vals])
                dense_centers = [uv2car_torch(torch.tensor([uv[0]], device="cuda"),
                        torch.tensor([uv[1]], device="cuda")) * sample_radius for uv in dense_uvs]

                candidate_cams, candidate_images = [], []
                for cam_center in proposal_centers:
                    dummy = DummyCamera(*look_at(cam_center.detach(), object_center.detach()), reference_camera)
                    rgb = render_with_oracle(cam_center, object_center, pipe, oracle_gaussians, background, reference_camera)
                    dummy.original_image = rgb.detach().clamp(0, 1).cuda()
                    render_pkg = render(dummy, gaussians, pipe, background)
                    candidate_cams.append(dummy)
                    candidate_images.append(render_pkg["render"])

                dense_cams = []
                for cam_center in dense_centers:
                    dummy = DummyCamera(*look_at(cam_center.detach(), object_center.detach()), reference_camera)
                    rgb = render_with_oracle(cam_center, object_center, pipe, oracle_gaussians, background, reference_camera)
                    dummy.original_image = rgb.detach().clamp(0, 1).cuda()
                    render_pkg = render(dummy, gaussians, pipe, background)
                    dense_cams.append(dummy)

                fisher_vals_ablation = selector.compute_fisher_uncertainty(gaussians, dense_cams, I_train_diag, pipe, background)
                
                if not args.deepkgp:
                    uncertainties = selector.compute_fisher_uncertainty(gaussians, candidate_cams, I_train_diag, pipe, background)

                    # Filter candidate poses that have been used before (by UV)
                    available_idxs = []
                    for idx, uv in enumerate(proposal_uvs):
                        uv_tuple = (round(uv[0], 5), round(uv[1], 5))  # round to reduce float precision issues
                        if uv_tuple not in used_uvs_global:
                            available_idxs.append(idx)

                    if not available_idxs:
                        print(f"[Warning] All candidate poses in sector {sector_id} already used. Skipping.")
                        continue

                    # Select among unused
                    available_uncertainties = uncertainties[available_idxs]
                    max_local_idx = torch.argmax(available_uncertainties).item()
                    max_global_idx = available_idxs[max_local_idx]

                    # Register UV as used
                    uv_tuple = (round(proposal_uvs[max_global_idx][0], 5), round(proposal_uvs[max_global_idx][1], 5))
                    used_uvs_global.add(uv_tuple)

                    # Final selection
                    final_cam = candidate_cams[max_global_idx]
                    oracle_img = render_with_oracle(final_cam.camera_center, object_center, pipe, oracle_gaussians, background, reference_camera)
                    final_cam.original_image = oracle_img.detach().clamp(0.0, 1.0).cuda()         
                elif args.vdgp:
                    pose_tensor = torch.stack([cam.camera_center for cam in candidate_cams], dim=0).float().to("cuda")  # (N, 3)
                    image_tensor = torch.stack(candidate_images, dim=0).float().to("cuda")  # (N, 3, H, W)

                    # Step 2: Train Φ for this sector
                    phi = train_phi_sector(pose_tensor, image_tensor)
                    phi.eval()
                    selector = VDGPFisherNBVSelector(
                        args,
                        input_dim=pose_feat_dim + image_feat_dim,  # total input dim to GP1
                        phi_pose_to_feat=phi,
                        device="cuda"
                    ) 
                    # Compute Fisher-trace based uncertainty at proposal poses
                    uncertainties = selector.compute_fisher_uncertainty(gaussians, selected_cams, candidate_cams, pipe, background)

                    # Select the most uncertain proposal
                    max_unc_idx = torch.argmax(uncertainties).item()
                    most_uncertain_uv = proposal_uvs[max_unc_idx]
                    # Apply small random perturbation
                    u_perturbed = most_uncertain_uv[0] + np.random.uniform(-0.02, 0.02)
                    v_perturbed = most_uncertain_uv[1] + np.random.uniform(-0.02, 0.02)

                    # Clamp to sector bounds
                    u_min, u_max = u_bounds
                    v_min, v_max = v_bounds

                    u_perturbed = np.clip(u_perturbed, u_min, u_max)
                    v_perturbed = np.clip(v_perturbed, v_min, v_max)

                    init_pose = (u_perturbed, v_perturbed)
                    sector_init_poses.append(proposal_centers[max_unc_idx].detach().cpu().numpy())
                    center_opt, uv_opt = selector.optimize_gp_posterior_vdgp(
                        proposal_uvs=[all_uvs[i] for i in sector_indices],
                        proposal_centers=[all_centers[i].cpu().numpy() for i in sector_indices],
                        uncertainties=uncertainties,  # <-- must be a torch.Tensor
                        init_uv=init_pose,
                        uv_bounds=(u_bounds, v_bounds),
                        radius=sample_radius,
                        object_center=object_center,  # <-- make sure this is passed
                        steps=args.pose_optim_steps,
                        lr=args.pose_lr
                    )
                    # Create DummyCamera for selected pose
                    oracle_img = render_with_oracle(center_opt, object_center, pipe, oracle_gaussians, background, reference_camera)
                    final_cam = DummyCamera(*look_at(center_opt.detach(), object_center.detach()), reference_camera, image=oracle_img.detach())
                elif args.deepkgp:
                    uncertainties = selector.compute_fisher_uncertainty(gaussians, candidate_cams, I_train_diag, pipe, background)
                    if train_iter > 1:
                        init_pose = get_robust_init_pose(proposal_uvs, uncertainties, u_bounds, v_bounds, sample_radius, strategy="diverse-fisher", eval_selected_cams=eval_selected_cams)
                    else:
                        init_pose = get_robust_init_pose(proposal_uvs, uncertainties, u_bounds, v_bounds, sample_radius, strategy="weighted", eval_selected_cams=eval_selected_cams)   
                    # # Find proposal pose closest to midpoint to record as init
                    # uv_dists = [np.linalg.norm(np.array(uv) - np.array(init_pose)) for uv in proposal_uvs]
                    # closest_idx = np.argmin(uv_dists)
                    # sector_init_poses.append(proposal_centers[closest_idx].detach().cpu().numpy())

                    # fisherrf_cam = candidate_cams[max_unc_idx]
                    # fisherrf_rendered = render(fisherrf_cam, gaussians, pipe, background)["render"].clamp(0, 1)

                    # np.save(os.path.join(sector_dir, "fisherrf_pose.npy"), fisherrf_cam.camera_center.cpu().numpy())
                    # TF.to_pil_image(fisherrf_rendered.cpu()).save(os.path.join(sector_dir, "fisherrf_image.png"))
                    image_encoder = ImageEncoder(output_dim=128).to("cuda")

                    center_opt, uv_opt, dense_uvs, acq_dense = selector.optimize_gp_posterior_dkl(
                        proposal_uvs=[all_uvs[i] for i in sector_indices],
                        proposal_centers=[all_centers[i].cpu().numpy() for i in sector_indices],
                        uncertainties=uncertainties,
                        dense_centers = dense_centers,
                        dense_uvs = dense_uvs,
                        init_uv=init_pose,
                        uv_bounds=(u_bounds, v_bounds),
                        radius=sample_radius,
                        object_center=object_center,
                        selected_cameras=eval_selected_cams,
                        gaussians=gaussians,
                        pipe=pipe,
                        background=background,
                        reference_camera=reference_camera,
                        render_fn=render_fn,
                        image_encoder=image_encoder,
                        steps=args.pose_optim_steps,
                        lr=args.pose_lr
                    )
                    # init_pose_tensor = torch.tensor(init_pose, dtype=torch.float32, device="cuda")
                    # u = init_pose_tensor[0].unsqueeze(0)  # shape [1]
                    # v = init_pose_tensor[1].unsqueeze(0)  # shape [1]
                    # center_opt = uv2car_torch(u, v).squeeze(0) * sample_radius
                    oracle_img = render_with_oracle(center_opt, object_center, pipe, oracle_gaussians, background, reference_camera)
                    deepkgp_rendered_img = render_fn(center_opt, object_center, pipe, gaussians, background, reference_camera)
                    final_cam = DummyCamera(*look_at(center_opt.detach(), object_center.detach()), reference_camera, image=oracle_img.detach())
                    
                    np.save(os.path.join(sector_dir, "deepkgp_pose.npy"), center_opt.cpu().numpy())
                    TF.to_pil_image(deepkgp_rendered_img.clamp(0, 1).cpu()).save(os.path.join(sector_dir, "deepkgp_image.png"))

                    def to_numpy(x):
                        if isinstance(x, torch.Tensor):
                            return x.detach().cpu().numpy()
                        return np.array(x)

                    np.savez(f"{args.model_path}/sector_{sector_id}_iter{train_iter}_acqmap.npz",
                                uvs=dense_uvs,
                                fisher_vals=to_numpy(fisher_vals_ablation),
                                acquisition=acq_dense)
                
                elif method == "random":
                    u_random = np.random.uniform(u_bounds[0], u_bounds[1])
                    v_random = np.random.uniform(v_bounds[0], v_bounds[1])
                    center_random = uv2car_torch(
                        torch.tensor([u_random], device="cuda"),
                        torch.tensor([v_random], device="cuda")
                    ).squeeze(0) * sample_radius
                    oracle_img = render_with_oracle(center_random, object_center, pipe, oracle_gaussians, background, reference_camera)
                    final_cam = DummyCamera(*look_at(center_random.detach(), object_center.detach()), reference_camera, image=oracle_img.detach())
                
                elif method == "middle":
                    u_middle = (u_bounds[0] + u_bounds[1]) / 2
                    v_middle = (v_bounds[0] + v_bounds[1]) / 2
                    center_middle = uv2car_torch(
                        torch.tensor([u_middle], device="cuda"),
                        torch.tensor([v_middle], device="cuda")
                    ).squeeze(0) * sample_radius
                    oracle_img = render_with_oracle(center_middle, object_center, pipe, oracle_gaussians, background, reference_camera)
                    final_cam = DummyCamera(*look_at(center_middle.detach(), object_center.detach()), reference_camera, image=oracle_img.detach())

                sector_selections.append(final_cam)
                img_path = f"oracle_gt_visualization/pose_{len(eval_selected_cams) + len(sector_selections)}.png"
                TF.to_pil_image(oracle_img.clamp(0, 1).cpu()).save(img_path)

            sector_selected_cams = sector_selections  # Only train on new NBVs
            eval_selected_cams += sector_selected_cams
            selected_cams = eval_selected_cams

            print(f"=== Training on {len(selected_cams)} views (sector round {train_iter}) ===")

            viewpoint_stack = selected_cams.copy()

            for iteration in tqdm(range(1, args.iterations + 1), desc=f"NBV Training Iter {train_iter}"):
                gaussians.update_learning_rate(iteration)

                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                if not viewpoint_stack:
                    viewpoint_stack = selected_cams.copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image = render_pkg["render"]
                gt_image = viewpoint_cam.original_image.cuda()

                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                loss.backward()

                with torch.no_grad():
                    if iteration < opt.densify_until_iter:
                        viewspace_point_tensor, visibility_filter, radii = (
                            render_pkg["viewspace_points"],
                            render_pkg["visibility_filter"],
                            render_pkg["radii"]
                        )
                        gaussians.max_radii2D[visibility_filter] = torch.max(
                            gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                        )
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.densify_and_prune(opt.densify_grad_threshold, args.min_opacity,
                                                        scene.cameras_extent, size_threshold)

                        if iteration % opt.opacity_reset_interval == 0 or (
                                dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()

                    if iteration < args.iterations:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)

            # Evaluate using fixed train+test cameras
            lpips_metric = lpips_func("cuda", net_type='vgg')
            psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
            test_cams = scene.getAllCameras(1.0)

            for i, cam in enumerate(uniform_test_cameras):
                rendered = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
                psnr_total += psnr(rendered, oracle_renders[i]).mean().item()
                ssim_total += ssim(rendered, oracle_renders[i]).mean().item()
                lpips_total += lpips_metric(rendered, oracle_renders[i]).mean().item()

            num_eval = len(uniform_test_cameras)
            avg_psnr = psnr_total / num_eval
            avg_ssim = ssim_total / num_eval
            avg_lpips = lpips_total / num_eval

            print(f"[Iter {train_iter}] PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
            log_metrics(log_path, train_iter, avg_psnr, avg_ssim, avg_lpips)

            # Initialize the metrics_curve list only once
            if not hasattr(args, "metrics_curve"):
                args.metrics_curve = []

            args.metrics_curve.append({
                "iteration": train_iter,
                "pose_lr": args.pose_lr,
                "pose_optim_steps": args.pose_optim_steps,
                "psnr": psnr_total / num_eval,
                "ssim": ssim_total / num_eval,
                "lpips": lpips_total / num_eval
            })
        output_dir = os.path.join("logs", "pose_opt_sweep", f"lr_{args.pose_lr}_steps_{args.pose_optim_steps}")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "curve.json"), "w") as f:
            json.dump(args.metrics_curve, f, indent=2)

def training_report(tb_writer, iteration, Ll1, loss, l1_loss_fn, elapsed, test_iterations, scene, render_fn, render_args):
    wandb.log({
        'train/l1_loss': Ll1.item(),
        'train/total_loss': loss.item(),
        'iter_time': elapsed
    }, step=iteration)

    if iteration in test_iterations:
        print(f"Running evaluation for iteration: {iteration}")
        torch.cuda.empty_cache()
        cams = scene.getAllCameras()
        l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0

        lpips_metric = lpips_func("cuda", net_type='vgg')
        for cam in cams:
            image = render_fn(cam, scene.gaussians, *render_args)["render"].clamp(0, 1)
            gt_image = cam.original_image.cuda().clamp(0, 1)
            l1_test += l1_loss_fn(image, gt_image).mean().item()
            psnr_test += psnr(image, gt_image).mean().item()
            ssim_test += ssim(image, gt_image).mean().item()
            lpips_test += lpips_metric(image, gt_image).mean().item()

        n = len(cams)
        log_dict = {
            'test/l1_loss': l1_test / n,
            'test/psnr': psnr_test / n,
            'test/ssim': ssim_test / n,
            'test/lpips': lpips_test / n,
            'total_points': scene.gaussians.get_xyz.shape[0]
        }
        wandb.log(log_dict, step=iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Sector-based Active Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--initial_train", type=int, default=5_000, help="Iterations for initial training")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--oracle_model_path", type=str, required=True, help="Path to pretrained 3DGS model for oracle rendering")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Flags for view selections
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--reg_lambda", type=float, default=1e-6)
    parser.add_argument("--I_test", action="store_true", help="Use I test to get the selection base")
    parser.add_argument("--I_acq_reg", action="store_true", help="apply reg_lambda to acq H too")
    parser.add_argument("--sh_up_every", type=int, default=5_000, help="increase spherical harmonics every N iterations")
    parser.add_argument("--sh_up_after", type=int, default=-1, help="start to increate active_sh_degree after N iterations")
    parser.add_argument("--min_opacity", type=float, default=0.005, help="min_opacity to prune")
    parser.add_argument("--filter_out_grad", nargs="+", type=str, default=["rotation"])
    parser.add_argument("--num_circles", type=int, default=5, help="Number of circles on the view-hemisphere, that contains proposal poses")
    parser.add_argument("--min_poses", type=int, default=30, help="Number of proposal poses on the smallest circle")
    parser.add_argument("--pose_lr", type=float, default=1e-4, help="Learning rate for pose optimization")
    parser.add_argument("--pose_optim_steps", type=float, default=200, help="Number of steps for pose optimization")
    parser.add_argument("--nbv_process", type=str, default="optimization", help="Process of reaching NBV", choices=["optimization", "selection"])
    parser.add_argument("--deepkgp", action="store_true", help="Use Deep Kernel GP for uncertainty approximation")
    parser.add_argument("--vdgp", action="store_true", help="Use Variational Deep GP for uncertainty approximation")
    parser.add_argument("--max_nbv_iterations", type=int, default=1, help="Iterations of Revolution around the object")
    parser.add_argument("--kernel_type", type=str, default="rbf", help="kernel type for deepkgp", choices=["tps", "rbf"])
    parser.add_argument("--exclude_deep_kernel", action="store_true", help="For remove deep feature extractor from DKL")
    args = parser.parse_args()

    # Define sweep values
    pose_lrs = [5e-4]
    pose_steps = [200]

    for lr in pose_lrs:
        for steps in pose_steps:
            print(f"\n=== Running training with LR={lr}, STEPS={steps} ===\n")

            # Update args dynamically
            args.pose_lr = lr
            args.pose_optim_steps = steps
            args.metrics_curve = []  # clear metrics from previous run

            wandb.init(project="active", config=vars(args), reinit=True)
            safe_state(False)

            training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args)

            # Save metrics
            output_dir = os.path.join("logs", "pose_opt_sweep", f"lr_{lr}_steps_{steps}")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "curve.json"), "w") as f:
                json.dump(args.metrics_curve, f, indent=2)

            wandb.finish()