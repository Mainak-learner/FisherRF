# active/sfm_uncertainty_selector.py

import numpy as np
from .uncertainty_sfm import evaluate_pose_uncertainty
from scene.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import rand_rotation_matrix
import torch
from typing import List

class SFMUncertaintySelector:
    def __init__(self, args):
        self.args = args
        self.num_perturbations = getattr(args, 'num_perturbations', 5)  # Default value
        self.deflection = getattr(args, 'pose_deflection', 0.1)
        self.translation_magnitude = getattr(args, 'translation_magnitude', 0.1)

    def nbvs(self, gaussians, scene, num_views, pipe, background, exit_func) -> List[int]:
        """
        Select next best views based on SfM-based uncertainty.
        """
        candidate_views = list(scene.get_candidate_set())
        candidate_cameras = scene.getCandidateCameras()  # Assume this method exists
        uncertainties = []

        for cam in candidate_cameras:
            # Evaluate uncertainty for this camera pose
            uncertainty = evaluate_pose_uncertainty(cam, gaussians, pipe, background,
                                                  num_perturbations=self.num_perturbations,
                                                  deflection=self.deflection,
                                                  translation_magnitude=self.translation_magnitude)
            uncertainties.append(uncertainty)

        # Select views with lowest uncertainty (most stable poses)
        sorted_indices = torch.argsort(uncertainties)[:num_views]
        selected_views = [candidate_views[i] for i in sorted_indices.tolist()]

        return selected_views