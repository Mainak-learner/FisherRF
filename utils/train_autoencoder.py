import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm
import numpy as np

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.camera_utils import look_at
from scene.cameras import DummyCamera
from utils.general_utils import safe_state

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def generate_oracle_images(output_dir, model_path, reference_camera, all_centers, background):
    os.makedirs(output_dir, exist_ok=True)
    oracle_gaussians = GaussianModel(sh_degree=3)  # Adjust if needed
    oracle_gaussians.load_ply(os.path.join(model_path, "point_cloud/iteration_30000/point_cloud.ply"))

    for i, cam_center in enumerate(tqdm(all_centers, desc="Rendering Oracle GT Images")):
        R, T = look_at(cam_center.detach(), oracle_gaussians.get_xyz.mean(dim=0).detach())
        dummy_cam = DummyCamera(R, T, reference_camera)
        img = render(dummy_cam, oracle_gaussians, None, background)["render"].clamp(0, 1)
        img_pil = transforms.ToPILImage()(img.cpu())
        img_pil.save(os.path.join(output_dir, f"pose_{i}.png"))

def train_autoencoder(image_dir, latent_dim=128, batch_size=32, epochs=20, lr=1e-3, device="cuda"):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        ToTensor()
    ])

    dataset = ImageDataset(image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for images in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            recon = model(images)
            loss = criterion(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), f"autoencoder_latent{latent_dim}.pth")

if __name__ == "__main__":
    image_dir = "oracle_gt_visualization"
    model_path = "oracle_model_path_here"
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    # Setup dummy dataset for scene info (just to extract reference_camera and pose info)
    dummy_scene = Scene(None, GaussianModel(3))  # assumes Scene doesn't fail on None dataset
    reference_camera = dummy_scene.getAllCameras()[0]

    # Load proposal pose centers
    all_centers = torch.tensor(np.load("oracle_gt_visualization/proposal_pose_centers.npy"), dtype=torch.float32, device="cuda")

    # Step 1: Generate images using oracle model
    generate_oracle_images(image_dir, model_path, reference_camera, all_centers, background)

    # Step 2: Train autoencoder
    train_autoencoder(image_dir=image_dir, latent_dim=128)
