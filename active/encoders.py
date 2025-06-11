import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, images):
        with torch.no_grad():
            x = self.backbone(images)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class PoseToImageEncoder(nn.Module):
    def __init__(self, pose_dim=3, image_feat_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_feat_dim)
        )

    def forward(self, poses):
        return self.net(poses)
    
class PoseToImageSIREN(nn.Module):
    def __init__(self, pose_dim=3, image_feat_dim=128, hidden_dim=256, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.fc1 = nn.Linear(pose_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, image_feat_dim)

    def forward(self, x):
        x = torch.sin(self.omega_0 * self.fc1(x))
        x = torch.sin(self.omega_0 * self.fc2(x))
        return self.fc3(x)
    
class FourierEncoder(nn.Module):
    def __init__(self, input_dim=3, num_frequencies=6):
        super().__init__()
        self.B = 2 ** torch.arange(num_frequencies).float() * 3.1415  # shape: [L]
        
    def forward(self, x):
        x = x.unsqueeze(-1) * self.B.to(x.device)  # [B, 3, L]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(x.shape[0], -1)  # [B, 3*2*L]

class PoseToImageFFMLP(nn.Module):
    def __init__(self, pose_dim=3, image_feat_dim=128, hidden_dim=256, num_frequencies=6):
        super().__init__()
        self.encoder = FourierEncoder(pose_dim, num_frequencies)
        self.net = nn.Sequential(
            nn.Linear(pose_dim * 2 * num_frequencies, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_feat_dim),
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        return self.net(x_encoded)