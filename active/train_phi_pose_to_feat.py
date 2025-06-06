import torch
import torch.nn as nn
from active.encoders import ImageEncoder, PoseToImageEncoder

def train_phi_sector(pose_tensor, image_tensor, device="cuda", pose_dim=3, image_feat_dim=128, epochs=300):
    """
    pose_tensor: (N, 3) candidate pose centers
    image_tensor: (N, 3, H, W) RGB images rendered at those poses
    Returns: trained PoseToImageEncoder model
    """
    image_encoder = ImageEncoder(output_dim=image_feat_dim).to(device)
    image_encoder.eval()

    with torch.no_grad():
        image_feats = image_encoder(image_tensor.to(device))  # (N, image_feat_dim)

    phi = PoseToImageEncoder(pose_dim=pose_dim, image_feat_dim=image_feat_dim).to(device)
    optimizer = torch.optim.Adam(phi.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    pose_tensor = pose_tensor.to(device)
    image_feats = image_feats.to(device)

    for epoch in range(epochs):
        pred_feats = phi(pose_tensor)
        loss = loss_fn(pred_feats, image_feats)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"[Φ Training] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return phi
