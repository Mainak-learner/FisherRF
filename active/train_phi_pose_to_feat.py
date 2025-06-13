import torch
import torch.nn as nn
from active.encoders import ImageEncoder, PoseToImageEncoder, PoseToImageSIREN, PoseToImageFFMLP

#To Prevent OOM issue:
def encode_images_in_batches(image_encoder, image_tensor, batch_size=4):
    feats = []
    with torch.no_grad():
        for i in range(0, image_tensor.shape[0], batch_size):
            batch = image_tensor[i:i+batch_size].to(image_tensor.device)
            feats.append(image_encoder(batch))
    return torch.cat(feats, dim=0)

def train_phi_sector(pose_tensor, image_tensor, device="cuda", pose_dim=3, image_feat_dim=128, epochs=300):
    """
    pose_tensor: (N, 3) candidate pose centers
    image_tensor: (N, 3, H, W) RGB images rendered at those poses
    Returns: trained PoseToImageEncoder model
    """
    image_encoder = ImageEncoder(output_dim=image_feat_dim).to(device)
    image_encoder.eval()

    image_feats = encode_images_in_batches(image_encoder=image_encoder, image_tensor=image_tensor)

    phi = PoseToImageSIREN(pose_dim=pose_dim, image_feat_dim=image_feat_dim, omega_0=30.0).to(device)
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
