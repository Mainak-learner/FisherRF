# lpips_selector.py
import torch
from lpipsPyTorch import lpips_func
import torchvision.transforms as T
import torch.nn.functional as F  # ensure this is at the top
from torch.optim import Adam
from tqdm import tqdm
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


    def optimize_pose(self, init_pose, render_fn, reference_imgs,
                    lr=1e-2, steps=100):
        u, v, r = [torch.tensor([v_], requires_grad=True, device=self.device, dtype=torch.float32)
                for v_ in init_pose]
        optimizer = Adam([u, v, r], lr=lr)

        for step in tqdm(range(steps), desc="Optimizing NBV"):
            optimizer.zero_grad()
            cam_center = uv2car_torch(u, v).to(self.device) * r
            rendered = render_fn(cam_center)  # returns image
            loss = -self.compute_lpips_loss(rendered, reference_imgs)  # maximize distance
            loss.backward(retain_graph=True)
            optimizer.step()

            # Enforce constraints
            v.data.clamp_(0.01, 0.49)
            u.data.remainder_(1.0)
            r.data.clamp_(3.0, 5.5)

        return u.item(), v.item(), r.item()
