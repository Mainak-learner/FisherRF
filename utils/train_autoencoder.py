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

    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        for images in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} (Latent {latent_dim})"):
            images = images.to(device)
            recon = model(images)
            loss = criterion(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Latent {latent_dim} | Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"autoencoder_latent{latent_dim}.pth")

    return best_loss

if __name__ == "__main__":
    image_dir = "oracle_gt_visualization"
    latent_dims = [16, 32, 64, 128, 256, 512]
    all_losses = {}

    for latent_dim in latent_dims:
        print(f"\n--- Training autoencoder with latent dim {latent_dim} ---")
        loss = train_autoencoder(image_dir=image_dir, latent_dim=latent_dim)
        all_losses[latent_dim] = loss

    print("\nSummary of reconstruction losses:")
    for dim, loss in all_losses.items():
        print(f"Latent Dim {dim}: Loss = {loss:.4f}")
