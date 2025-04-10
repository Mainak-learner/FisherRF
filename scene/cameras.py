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
                 uid=-1, data_device="cuda", depth=None):
        self.colmap_id = colmap_id
        self.R = torch.from_numpy(R).float().to(data_device)
        self.T = torch.from_numpy(T).float().to(data_device)
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.uid = uid
        self.data_device = data_device

        # Handle image=None case explicitly
        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        else:
            self.original_image = None  # No ground truth for novel poses

        self.gt_alpha_mask = gt_alpha_mask.to(self.data_device) if gt_alpha_mask is not None else None
        self.depth = depth.to(self.data_device) if depth is not None else None

        # Set default height and width if no image
        self.height = self.original_image.shape[1] if self.original_image is not None else 512
        self.width = self.original_image.shape[2] if self.original_image is not None else 512

        self.FovX = FoVx
        self.FovY = FoVy
        self.znear = 0.01
        self.zfar = 100.0

        self.world_view_transform = torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]).to(self.data_device).float() @ torch.cat([self.R, self.T[:, None]], 1)

        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FovX, fovY=self.FovY
        ).transpose(0, 1).to(self.data_device).float()

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).transpose(1, 0) @ 
                                  self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = -torch.bmm(self.R.transpose(0, 1).unsqueeze(0), self.T[:, None])[:, 0]

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

