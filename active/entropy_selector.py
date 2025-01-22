import torch
from tqdm import tqdm
from typing import List
from scene import Scene
from gaussian_renderer import modified_render

class EntropySelector(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # Initialize any parameters needed for your entropy calculation
        self.num_candidates = getattr(args, 'num_candidates', 100)
        self.seed = getattr(args, 'seed', 0)
        self.debug = getattr(args, 'debug', False)
        self.views_added = 0

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
        selected_views = [candidate_views[i] for i in selected_indices.tolist()]
        
        # Update views added
        self.views_added += len(selected_views)
        
        return selected_views
        
        
