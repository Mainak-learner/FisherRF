import torch
from tqdm import tqdm
from typing import List
from scene import Scene
from gaussian_renderer import modified_render

class EntropySelector(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # Initialize any parameters needed for your entropy calculation

    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, exit_func) -> List[int]:
        candidate_views = list(scene.get_candidate_set())
        candidate_cameras = scene.getCandidateCameras()
        
        entropy_scores = []

        for cam in tqdm(candidate_cameras, desc="Calculating entropy for candidate views"):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            entropy = render_pkg["entropy"]
            entropy_scores.append(entropy.mean().item())  # Use mean entropy across the image


        # Select the views with the highest entropy scores
        selected_indices = torch.tensor(entropy_scores).argsort(descending=True)[:num_views]
        return [candidate_views[i] for i in selected_indices.tolist()]
        
        
