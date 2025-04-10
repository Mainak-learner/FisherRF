#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask=None, image_name="",
                 uid=-1, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", depth=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = torch.from_numpy(R).float().to(data_device)  # Ensure 3x3
        self.T = torch.from_numpy(T).float().to(data_device).view(3, 1)  # Ensure 3x1
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        # Handle image=None case
        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        else:
            self.original_image = None
            self.image_width = 512  # Default width
            self.image_height = 512  # Default height

        if gt_alpha_mask is not None and self.original_image is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        elif self.original_image is not None:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = torch.from_numpy(trans).float().to(self.data_device)
        self.scale = scale

        # Compute world_view_transform using getWorld2View2
        self.world_view_transform = torch.tensor(
            getWorld2View2(self.R.cpu().numpy(), self.T.cpu().numpy(), self.trans.cpu().numpy(), self.scale)
        ).transpose(0, 1).to(self.data_device)

        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).to(self.data_device)

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.depth = depth.to(self.data_device) if depth is not None else None

    # Add a safeguard method to prevent accidental access
    def get_image_or_none(self):
        return self.original_image if self.original_image is not None else torch.zeros(3, self.height, self.width).to(self.data_device)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

