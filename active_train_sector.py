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
from utils.camera_utils import look_at
from scene.cameras import DummyCamera
from arguments import ModelParams, PipelineParams, OptimizationParams


def render_fn(cam_center, object_center, pipe, gaussians, background, reference_camera):
    R, T = look_at(cam_center, object_center)
    dummy_cam = DummyCamera(R, T, reference_camera)
    return render(dummy_cam, gaussians, pipe, background)["render"]


def render_with_oracle(cam_center, object_center, pipe, oracle_gaussians, background, reference_camera):
    R, T = look_at(cam_center, object_center)
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
    Scene.getCamera = lambda self, idx, scale=1.0: self.train_cameras[scale][idx]
    gaussians.training_setup(opt)

    oracle_gaussians = GaussianModel(dataset.sh_degree)
    oracle_gaussians.load_ply(os.path.join(args.oracle_model_path, "point_cloud/iteration_30000/point_cloud.ply"))

    object_center = oracle_gaussians.get_xyz.mean(dim=0).detach()
    reference_camera = scene.getAllCameras()[0]
    sample_radius = torch.norm(reference_camera.camera_center - object_center).item()

    all_centers, all_uvs = generate_circular_hemisphere_poses(object_center, radius=sample_radius)
    circle_indices, middle_circle_indices, sector_map = divide_hemisphere_poses(all_centers, object_center.cpu().numpy())

    # Train on middle circle views first
    middle_ids = np.random.choice(middle_circle_indices, size=6, replace=False)
    custom_train_indices = []
    for idx in middle_ids:
        cam_center = all_centers[idx]
        gt_img = render_with_oracle(cam_center, object_center, pipe, oracle_gaussians, torch.tensor([1.0, 1.0, 1.0], device="cuda"), reference_camera)
        dummy_camera = DummyCamera(*look_at(cam_center, object_center), reference_camera, image=gt_img.detach())
        scene.train_cameras[1.0].append(dummy_camera)
        custom_train_indices.append(len(scene.train_cameras[1.0]) - 1)

    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    # Train the untrained model with 6 middle circle views first
    for iteration in tqdm(range(1, 5001), desc="Initial Training on Middle Circle"):
        gaussians.update_learning_rate(iteration)
        cam_idx = custom_train_indices[randint(0, len(custom_train_indices)-1)]
        viewpoint_cam = scene.getCamera(cam_idx)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

    # Prepare LPIPS selector
    selector = LPIPSNBVSelector()

    for sector_id, sector_indices in sector_map.items():
        if len(sector_indices) == 0:
            continue

        sector_centers = all_centers[sector_indices]
        mean_center = sector_centers.mean(0)
        dir_vec = mean_center - object_center
        radius = torch.norm(dir_vec).item()
        dir_vec /= radius

        u = (np.arctan2(dir_vec[1].item(), dir_vec[0].item()) / (2 * np.pi)) % 1.0
        v = np.arccos(dir_vec[2].item()) / np.pi
        init_pose = (u, v, radius)

        sector_ref_imgs = []
        for idx in sector_indices:
            cam_center = all_centers[idx]
            img = render_fn(cam_center, object_center, pipe, gaussians, background, reference_camera)
            sector_ref_imgs.append(img)

        u_opt, v_opt, r_opt = selector.optimize_pose(
            init_pose,
            lambda cam: render_fn(cam, object_center, pipe, gaussians, background, reference_camera),
            sector_ref_imgs
        )
        new_cam_center = selector.uv_to_xyz(torch.tensor([u_opt]), torch.tensor([v_opt])) * r_opt + object_center
        oracle_img = render_with_oracle(new_cam_center, object_center, pipe, oracle_gaussians, background, reference_camera)

        dummy_camera = DummyCamera(*look_at(new_cam_center, object_center), reference_camera, image=oracle_img.detach())
        scene.train_cameras[1.0].append(dummy_camera)
        new_idx = len(scene.train_cameras[1.0]) - 1
        custom_train_indices.append(new_idx)

    print(f"Final selected training views: {custom_train_indices}")

    lpips_metric = lpips_func("cuda", net_type='vgg')
    total_iterations = args.iterations
    full_training_iters = total_iterations - 5000

    for iteration in tqdm(range(1, full_training_iters + 1), desc="Full Training Loop"):
        current_iter = 5000 + iteration
        gaussians.update_learning_rate(current_iter)
        cam_idx = custom_train_indices[randint(0, len(custom_train_indices)-1)]
        viewpoint_cam = scene.getCamera(cam_idx)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        if current_iter in test_iterations:
            test_cams = scene.getTestCameras()
            psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
            for cam in test_cams:
                test_img = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
                gt_img = cam.original_image.cuda().clamp(0, 1)
                psnr_total += psnr(test_img, gt_img).item()
                ssim_total += ssim(test_img, gt_img).item()
                lpips_total += lpips_metric(test_img, gt_img).mean().item()
            num = len(test_cams)
            print(f"[ITER {current_iter}] PSNR {psnr_total/num:.2f} SSIM {ssim_total/num:.4f} LPIPS {lpips_total/num:.4f}")

        if current_iter in save_iterations:
            scene.save(current_iter)

    return custom_train_indices





if __name__ == "__main__":
    parser = ArgumentParser(description="Sector-based Active Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15000, 20000, 25000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--oracle_model_path", type=str, required=True, help="Path to pretrained 3DGS model for oracle rendering")
    args = parser.parse_args()

    wandb.init(project="active", config=vars(args))
    safe_state(False)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args)
    wandb.finish()
