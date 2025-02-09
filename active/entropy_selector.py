import torch
from tqdm import tqdm
from typing import List
from scene import Scene
import numpy as np
import random
from gaussian_renderer import modified_render

def calculate_distance(cam1, cam2):
    return torch.norm(cam1.camera_center - cam2.camera_center)

class EntropySelector(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # Initialize any parameters needed for your entropy calculation
        self.num_candidates = getattr(args, 'num_candidates', 100)
        self.seed = getattr(args, 'seed', 0)
        self.debug = getattr(args, 'debug', False)
        self.distance_sigma = args.distance_sigma  # Controls the influence of distance
        self.views_added = 0
        self.use_scheduler = args.use_scheduler

    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, completion_rate, exit_func) -> List[int]:
        candidate_views = list(scene.get_candidate_set())
        candidate_cameras = scene.getCandidateCameras()
        train_cameras = scene.getTrainCameras()
        
        entropy_scores = []
        distance_weights = []
        min_distances = []

        if self.use_scheduler:
            new_sigma = self.distance_sigma
        else:
            new_sigma = self.update_distance_sigma(completion_rate)

        for cam in tqdm(candidate_cameras, desc="Calculating entropy for candidate views"):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            entropy = render_pkg["entropy"]
            entropy_scores.append(entropy.mean().item())

            # Calculate minimum distance to existing training views
            min_distance = min(calculate_distance(cam, train_cam) for train_cam in train_cameras)
            min_distances.append(min_distance)
            distance_weight = torch.exp(-min_distance / new_sigma)
            distance_weights.append(distance_weight)
        
        max_min_dist = max(min_distances)

        #Boltzmann exploration (max-min distance is the temperature):
        probs = np.exp(np.array(entropy_scores) / max_min_dist.item())

        # if torch.sigmoid((max(min_distances) - min(min_distances))/min(min_distances)) < 0.6:
        #     weighted_scores = entropy_scores
        # else:
        #     weighted_scores = [e * w for e, w in zip(entropy_scores, distance_weights)]

        # # Select the views with the highest weighted entropy scores
        # selected_indices = torch.tensor(weighted_scores).argsort(descending=True)[:num_views]

        # return [candidate_views[i] for i in selected_indices.tolist()]

        #Select based on the boltzmann distribution:
        selected_indices = np.random.choice(list(range(len(probs))), size=num_views, replace=False, p=(probs/np.sum(probs)))        

        return [candidate_views[i] for i in selected_indices]

    def update_distance_sigma(self, completion_rate):
        return self.distance_sigma * 0.5 * (1 + np.cos(np.pi * completion_rate))         
        
