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
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image=None, gt_alpha_mask=None, image_name="", uid=-1, 
                 data_device="cuda", height=None, width=None):
        self.colmap_id = colmap_id
        self.R = torch.from_numpy(R).float().to(data_device)  # Rotation matrix
        self.T = torch.from_numpy(T).float().to(data_device)  # Translation vector
        self.FoVx = FoVx  # Field of view in x direction
        self.FoVy = FoVy  # Field of view in y direction
        self.image_name = image_name
        self.uid = uid
        self.data_device = data_device

        # Handle image and dimensions
        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_height = self.original_image.shape[1]
            self.image_width = self.original_image.shape[2]

            # Apply gt_alpha_mask if provided, otherwise use a ones mask
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.clamp(0.0, 1.0).to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        else:
            self.original_image = None  # No ground truth image for perturbed poses
            if height is None or width is None:
                raise ValueError("Height and width must be provided if no image is given for the camera.")
            self.image_height = height
            self.image_width = width

        # Store gt_alpha_mask (can be None)
        self.gt_alpha_mask = gt_alpha_mask.clamp(0.0, 1.0).to(self.data_device) if gt_alpha_mask is not None else None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

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

