# lpips_selector.py
import torch
from lpipsPyTorch import lpips_func
import torchvision.transforms as T
import torch.nn.functional as F  # ensure this is at the top
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from utils.graphics_utils import uv2car_torch

class LPIPSNBVSelector:
    def __init__(self, device='cuda'):
        self.lpips_model = lpips_func("cuda", net_type='vgg')
        self.device = device
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def compute_lpips_loss(self, E_p, reference_imgs):
        if isinstance(E_p, torch.Tensor):
            E_p_tensor = E_p.unsqueeze(0).to(self.device)
        else:
            E_p_tensor = self.transform(E_p).unsqueeze(0).to(self.device)

        # ⬇️ Downsample to 64×64 for memory efficiency
        E_p_tensor = F.interpolate(E_p_tensor, size=(64, 64), mode='bilinear', align_corners=False)

        total_lpips = 0.0
        for ref_img in reference_imgs:
            if isinstance(ref_img, torch.Tensor):
                ref_tensor = ref_img.unsqueeze(0).to(self.device)
            else:
                ref_tensor = self.transform(ref_img).unsqueeze(0).to(self.device)

            ref_tensor = F.interpolate(ref_tensor, size=(64, 64), mode='bilinear', align_corners=False)  # ⬅️ Add this
            lpips_val = self.lpips_model(E_p_tensor, ref_tensor)
            total_lpips += lpips_val
        return total_lpips / len(reference_imgs)


    def optimize_pose(self, init_pose, render_fn, reference_imgs, sector_indices, all_uvs, sample_radius,
                      lr=1e-2, steps=100):
        """
        init_pose: tuple of (u, v, r)
        render_fn: function(cam_center) -> image
        reference_imgs: list of rendered images from sector
        sector_indices: np.array of indices into all_centers for this sector
        all_centers: torch.Tensor of shape (150, 3) (centered at origin)
        object_center: torch.Tensor of shape (3,)
        """

        # Optimize u, v, r
        u, v, r = [torch.tensor([x], requires_grad=True, dtype=torch.float32, device=self.device)
                   for x in init_pose]
        optimizer = Adam([u, v, r], lr=lr)

        # Compute clamping range from sector
        sector_uvs = [all_uvs[i] for i in sector_indices]
        u_vals = [uv[0] for uv in sector_uvs]
        v_vals = [uv[1] for uv in sector_uvs]

        margin = 0.01
        u_min = min(u_vals)
        u_max = max(u_vals)
        v_min = min(v_vals)
        v_max = max(v_vals)
        r_min = sampled_radius - margin
        r_max = sampled_radius + margin

        for _ in tqdm(range(steps), desc="Optimizing NBV"):
            optimizer.zero_grad()
            cam_center = uv2car_torch(u, v) * r  # Positioned around origin
            rendered = render_fn(cam_center)
            loss = -self.compute_lpips_loss(rendered, reference_imgs)  # Maximize LPIPS
            loss.backward()
            optimizer.step()

            u.data.clamp_(u_min, u_max)
            v.data.clamp_(v_min, v_max)
            r.data.clamp_(r_min, r_max)

        return u.item(), v.item(), r.item()
