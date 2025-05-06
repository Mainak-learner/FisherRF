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
from active.lpips_optim_selector import LPIPSNBVSelector
from scene.pose_sampler import divide_hemisphere_poses
from utils.graphics_utils import look_at
from scene.cameras import DummyCamera
from arguments import ModelParams, PipelineParams, OptimizationParams


def render_fn(cam_center, object_center, pipe, gaussians, background):
    R, T = look_at(cam_center, object_center)
    dummy_cam = DummyCamera(R, T)
    return render(dummy_cam, gaussians, pipe, background)["render"]

def render_with_oracle(cam_center, object_center, pipe, oracle_gaussians, background):
    R, T = look_at(cam_center, object_center)
    dummy_cam = DummyCamera(R, T)
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

    # Load pretrained model as oracle
    oracle_gaussians = GaussianModel(dataset.sh_degree)
    oracle_gaussians.load_ply(os.path.join(args.oracle_model_path, "point_cloud/iteration_30000/point_cloud.ply"))

    object_center = gaussians.get_xyz.mean(dim=0).detach()
    all_poses = scene.getAllCameras()
    all_centers = torch.stack([cam.camera_center for cam in all_poses]).cpu().numpy()
    circle_indices, middle_circle_indices, sector_map = divide_hemisphere_poses(all_centers, object_center.cpu().numpy())

    middle_ids = np.random.choice(middle_circle_indices, size=6, replace=False)
    scene.train_idxs.extend(list(middle_ids))

    ref_imgs = []
    for idx in scene.train_idxs:
        cam = scene.getCamera(idx)
        img = render_fn(cam.camera_center, object_center, pipe, gaussians, torch.tensor([1.0, 1.0, 1.0], device="cuda"))
        ref_imgs.append(img)

    selector = LPIPSNBVSelector()

    for sector_id, sector_indices in sector_map.items():
        if len(sector_indices) == 0:
            continue

        sector_centers = torch.stack([scene.getCamera(idx).camera_center for idx in sector_indices], dim=0)
        mean_center = sector_centers.mean(0)
        dir_vec = mean_center - object_center
        radius = torch.norm(dir_vec).item()
        dir_vec /= radius

        u = (np.arctan2(dir_vec[1].item(), dir_vec[0].item()) / (2 * np.pi)) % 1.0
        v = np.arccos(dir_vec[2].item()) / np.pi
        init_pose = (u, v, radius)

        u_opt, v_opt, r_opt = selector.optimize_pose(
            init_pose,
            lambda cam: render_fn(cam, object_center, pipe, gaussians, torch.tensor([1.0, 1.0, 1.0], device="cuda")),
            ref_imgs
        )
        new_cam_center = selector.uv_to_xyz(torch.tensor([u_opt]), torch.tensor([v_opt])) * r_opt + object_center
        oracle_img = render_with_oracle(new_cam_center, object_center, pipe, oracle_gaussians, torch.tensor([1.0, 1.0, 1.0], device="cuda"))
        ref_imgs.append(oracle_img)

        dummy_camera = DummyCamera(*look_at(new_cam_center, object_center))
        dummy_camera.original_image = oracle_img.detach()  # Assign GT image from oracle
        scene.train_cameras[1.0].append(dummy_camera)
        scene.all_train_set.add(len(scene.train_cameras[1.0]) - 1)
        scene.train_idxs.append(len(scene.train_cameras[1.0]) - 1)

    print(f"Final selected training views: {scene.train_idxs}")

    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    lpips_metric = lpips_func("cuda", net_type='vgg')

    for iteration in tqdm(range(1, args.iterations + 1), desc="Training loop"):
        gaussians.update_learning_rate(iteration)
        cam_idx = scene.train_idxs[randint(0, len(scene.train_idxs)-1)]
        viewpoint_cam = scene.getCamera(cam_idx)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration in test_iterations:
            test_cams = scene.getTestCameras()
            psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
            for cam in test_cams:
                test_img = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
                gt_img = cam.original_image.cuda().clamp(0, 1)
                psnr_total += psnr(test_img, gt_img).item()
                ssim_total += ssim(test_img, gt_img).item()
                lpips_total += lpips_metric(test_img, gt_img).mean().item()
            num = len(test_cams)
            print(f"[ITER {iteration}] PSNR {psnr_total/num:.2f} SSIM {ssim_total/num:.4f} LPIPS {lpips_total/num:.4f}")

        if iteration in save_iterations:
            scene.save(iteration)

    return scene.train_idxs


if __name__ == "__main__":
    parser = ArgumentParser(description="Sector-based Active Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--model_path", type=str, default="output/active_sector")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15000, 20000, 25000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--oracle_model_path", type=str, required=True, help="Path to pretrained 3DGS model for oracle rendering")
    args = parser.parse_args()

    wandb.init(project="active", config=vars(args))
    safe_state(False)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args)
    wandb.finish()
