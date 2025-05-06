# lpips_selector.py
import torch
import lpips
import torchvision.transforms as T
from torch.optim import Adam

class LPIPSNBVSelector:
    def __init__(self, device='cuda'):
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.device = device
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def compute_lpips_loss(self, E_p, reference_imgs):
        E_p_tensor = self.transform(E_p).unsqueeze(0).to(self.device)
        total_loss = 0.0
        for ref_img in reference_imgs:
            ref_tensor = self.transform(ref_img).unsqueeze(0).to(self.device)
            loss = self.lpips_model(E_p_tensor, ref_tensor)
            total_loss += loss
        return total_loss / len(reference_imgs)

    def optimize_pose(self, init_pose, render_fn, reference_imgs,
                      lr=1e-2, steps=100):
        # init_pose = (u, v, r)
        u, v, r = [torch.tensor([v_], requires_grad=True, device=self.device) for v_ in init_pose]
        optimizer = Adam([u, v, r], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            cam_center = self.uv_to_xyz(u, v) * r
            rendered = render_fn(cam_center)  # returns image
            loss = -self.compute_lpips_loss(rendered, reference_imgs)  # maximize distance
            loss.backward()
            optimizer.step()

            # enforce constraints (upper hemisphere, bounds)
            v.data.clamp_(0.01, 0.49)
            u.data.remainder_(1.0)
            r.data.clamp_(3.0, 5.5)

        return u.item(), v.item(), r.item()

    def uv_to_xyz(self, u, v):
        u = u * 2 * torch.pi
        v = v * torch.pi
        x = torch.cos(u) * torch.sin(v)
        y = torch.sin(u) * torch.sin(v)
        z = torch.cos(v)
        return torch.stack([x, y, z], dim=-1).squeeze(0)
