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
from scene.sector_pose_gen import generate_circular_hemisphere_poses, divide_hemisphere_poses
from utils.camera_utils import look_at, look_at_torch
from utils.graphics_utils import uv2car_torch
from scene.cameras import DummyCamera
from torchvision.utils import save_image
import base64
import json
from PIL import Image
from active.gp_predictor import GPFisherNBVSelector
import torchvision.transforms.functional as TF
from arguments import ModelParams, PipelineParams, OptimizationParams


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

def training(dataset, opt, pipe, test_iterations, save_iterations, args):
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    oracle_gaussians = GaussianModel(dataset.sh_degree)
    oracle_gaussians.load_ply(os.path.join(args.oracle_model_path, "point_cloud/iteration_30000/point_cloud.ply"))

    object_center = oracle_gaussians.get_xyz.mean(dim=0).detach()
    reference_camera = scene.getAllCameras()[0]
    sample_radius = torch.norm(reference_camera.camera_center).item()

    all_centers, all_uvs, pose_per_circle = generate_circular_hemisphere_poses(torch.tensor([0, 0, 0], device=object_center.device), radius=sample_radius)
    circle_indices, middle_circle_indices, sector_map = divide_hemisphere_poses(all_centers, object_center.cpu().numpy(), pose_per_circle)

    middle_ids = np.random.choice(middle_circle_indices, size=6, replace=False)
    selected_cams = []

    os.makedirs("oracle_gt_visualization", exist_ok=True)
    np.save("oracle_gt_visualization/object_center.npy", object_center.cpu().numpy())
    np.save("oracle_gt_visualization/object_points.npy", oracle_gaussians.get_xyz.detach().cpu().numpy())
    np.save("oracle_gt_visualization/object_colors.npy", oracle_gaussians._features_dc.detach().cpu().numpy())

    oracle_image_paths = []
    selected_middle_centers = []
    for i,idx in enumerate(middle_ids):
        cam_center = all_centers[idx]
        gt_img = render_with_oracle(cam_center, object_center, pipe, oracle_gaussians, torch.tensor([1.0, 1.0, 1.0], device="cuda"), reference_camera)
        img_path = f"oracle_gt_visualization/pose_{i}.png"
        TF.to_pil_image(gt_img.clamp(0, 1).cpu()).save(img_path)

        # Encode to base64
        oracle_image_paths.append(f"pose_{i}.png")
        dummy_camera = DummyCamera(*look_at(cam_center.detach(), object_center.detach()), reference_camera, image=gt_img.detach())
        selected_middle_centers.append(cam_center.cpu().numpy().tolist())
        selected_cams.append(dummy_camera)

    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

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
    
    lpips = lpips_func("cuda", net_type='vgg')
    psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
    os.makedirs("middle_render_vs_gt", exist_ok=True)
    for idx, cam in enumerate(selected_cams):
        rendered = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
        gt_image = cam.original_image.clamp(0, 1).to(rendered.device)  # ensure both are on the same device

        psnr_total += psnr(rendered, gt_image).mean().item()
        ssim_total += ssim(rendered, gt_image).mean().item()
        lpips_total += lpips(rendered, gt_image).mean().item()

        # Save copies for inspection
        save_image(rendered.cpu(), f"middle_render_vs_gt/render_{idx}.png")
        save_image(gt_image.cpu(), f"middle_render_vs_gt/gt_{idx}.png")
    
    N = len(selected_cams)
    print(f"[Middle Circle] PSNR: {psnr_total/N:.2f}, SSIM: {ssim_total/N:.4f}, LPIPS: {lpips_total/N:.4f}")
    selector = GPFisherNBVSelector(args, device="cuda")
    sector_selections = []
    for sector_id, sector_indices in sector_map.items():
        if len(sector_indices) == 0:
            continue
        
        sector_centers = all_centers[sector_indices]
        mean_center = sector_centers.mean(0)
        dir_vec = mean_center
        radius = torch.norm(dir_vec).item()
        dir_vec /= radius

        u = (np.arctan2(dir_vec[1].item(), dir_vec[0].item()) / (2 * np.pi)) % 1.0
        v = np.arccos(dir_vec[2].item()) / np.pi
        init_pose = (u, v, radius)

        # Get proposal UVs and centers for this sector
        proposal_uvs = [all_uvs[idx] for idx in sector_indices]
        proposal_centers = all_centers[sector_indices]

        u_vals = [uv[0] for uv in proposal_uvs]
        v_vals = [uv[1] for uv in proposal_uvs]

        u_bounds = (min(u_vals), max(u_vals))
        v_bounds = (min(v_vals), max(v_vals))

        candidate_cams = [DummyCamera(*look_at(cam_center.detach(), object_center.detach()), reference_camera) for cam_center in proposal_centers]

        # Compute Fisher-trace based uncertainty at proposal poses
        uncertainties = selector.compute_fisher_uncertainty(gaussians, selected_cams, candidate_cams, pipe, background)

        # sector_ref_imgs = []
        # for idx in sector_indices:
        #     cam_center = all_centers[idx]
        #     img = render_with_oracle(cam_center, object_center, pipe, gaussians, background, reference_camera)
        #     sector_ref_imgs.append(img)

        # u_opt, v_opt, r_opt = selector.optimize_pose(
        #     init_pose,
        #     lambda cam: render_fn(cam, object_center, pipe, gaussians, background, reference_camera, debug=True),
        #     sector_ref_imgs,
        #     sector_indices,
        #     all_uvs, 
        #     sample_radius
        # )
        # new_cam_center = uv2car_torch(torch.tensor([u_opt], device=object_center.device), torch.tensor([v_opt], device=object_center.device)) * r_opt
        # oracle_img = render_with_oracle(new_cam_center, object_center, pipe, oracle_gaussians, background, reference_camera)

        # Fit GP on proposal UVs and predict over full hemisphere
        center_opt, uv_opt = selector.optimize_gp_posterior(
            proposal_uvs=[all_uvs[i] for i in sector_indices],
            proposal_centers=[all_centers[i].cpu().numpy() for i in sector_indices],
            uncertainties=uncertainties,  # should be tensor
            init_uv=init_pose[:2],
            uv_bounds=(u_bounds, v_bounds),
            radius=sample_radius,
            steps=100
        )
        # Create DummyCamera for selected pose
        oracle_img = render_with_oracle(center_opt, object_center, pipe, oracle_gaussians, background, reference_camera)
        dummy_camera = DummyCamera(*look_at(center_opt.detach(), object_center.detach()), reference_camera, image=oracle_img.detach())

        sector_selections.append(dummy_camera)
        img_path = f"oracle_gt_visualization/pose_{len(sector_selections) + 6}.png"
        TF.to_pil_image(oracle_img.clamp(0, 1).cpu()).save(img_path)

        # dummy_camera = DummyCamera(*look_at(new_cam_center.detach(), object_center.detach()), reference_camera, image=oracle_img.detach())
        # selected_cams.append(dummy_camera)
        # img_path = f"oracle_gt_visualization/pose_{len(selected_cams)-1}.png"
        # TF.to_pil_image(oracle_img.clamp(0, 1).cpu()).save(img_path)
    
    print(f"Sector selections: {len(sector_selections)}")
    selected_cams = selected_cams + sector_selections
    print(f"Selected 18 training views. Final phase of training begins now...")

    filenames = [f"oracle_gt_visualization/pose_{i}.png" for i in range(len(selected_cams))]
    with open("oracle_gt_visualization/image_filenames.json", "w") as f:
        json.dump([os.path.basename(p) for p in filenames], f)
    pose_centers = torch.stack([cam.camera_center for cam in selected_cams], dim=0).cpu().numpy()
    np.save("oracle_gt_visualization/pose_centers.npy", pose_centers)

    for iteration in tqdm(range(1, args.iterations + 1), desc="Full Training Loop"):
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
            if iteration < args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

        if iteration in test_iterations:
            test_cams = scene.getAllCameras(1.0)
            psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
            test_save_path = f"test_inspection/iter_{iteration}"
            os.makedirs(test_save_path, exist_ok=True)
            for idx, cam in enumerate(test_cams):
                test_img = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
                gt_img = cam.original_image.cuda().clamp(0, 1)
                psnr_total += psnr(test_img, gt_img).mean().item()
                ssim_total += ssim(test_img, gt_img).mean().item()
                lpips_total += lpips(test_img, gt_img).mean().item()
                save_image(test_img.cpu(), os.path.join(test_save_path, f"render_{idx}.png"))
                save_image(gt_img.cpu(), os.path.join(test_save_path, f"gt_{idx}.png"))
            num = len(test_cams)
            print(f"[ITER {iteration}] PSNR {psnr_total/num:.2f} SSIM {ssim_total/num:.4f} LPIPS {lpips_total/num:.4f}")

        if iteration in save_iterations:
            scene.save(iteration)

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

        for cam in cams:
            image = render_fn(cam, scene.gaussians, *render_args)["render"].clamp(0, 1)
            gt_image = cam.original_image.cuda().clamp(0, 1)
            l1_test += l1_loss_fn(image, gt_image).mean().item()
            psnr_test += psnr(image, gt_image).mean().item()
            ssim_test += ssim(image, gt_image).mean().item()
            lpips_test += lpips(image, gt_image).mean().item()

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
    parser.add_argument("--initial_train", type=int, default=5000, help="Iterations for initial training")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5000, 10000, 15000, 20000, 25000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--oracle_model_path", type=str, required=True, help="Path to pretrained 3DGS model for oracle rendering")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Flags for view selections
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reg_lambda", type=float, default=1e-6)
    parser.add_argument("--I_test", action="store_true", help="Use I test to get the selection base")
    parser.add_argument("--I_acq_reg", action="store_true", help="apply reg_lambda to acq H too")
    parser.add_argument("--sh_up_every", type=int, default=5_000, help="increase spherical harmonics every N iterations")
    parser.add_argument("--sh_up_after", type=int, default=-1, help="start to increate active_sh_degree after N iterations")
    parser.add_argument("--min_opacity", type=float, default=0.005, help="min_opacity to prune")
    parser.add_argument("--filter_out_grad", nargs="+", type=str, default=["rotation"])
    args = parser.parse_args()

    wandb.init(project="active", config=vars(args))
    safe_state(False)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args)
    wandb.finish()
