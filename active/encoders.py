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