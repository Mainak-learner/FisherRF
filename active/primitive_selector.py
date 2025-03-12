import torch
from tqdm import tqdm
from gaussian_renderer import modified_render
from scene import Scene

class PrimitiveSelector(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.uncertainty_boost_new = 2.0  # Weight for new splats
        self.local_percentile = 0.1       # Top 10% uncertain pixels
        self.decay_factor = 0.95          # Temporal decay rate

    def compute_uncertainty_score(self, uncertainty_map):
        """Prioritize localized high uncertainty regions"""
        flattened = uncertainty_map.flatten()
        k = int(self.local_percentile * flattened.numel())
        return flattened.topk(k).values.mean()

    def nbvs(self, gaussians, scene, num_views, pipe, background, exit_func=None):
        candidate_views = list(scene.get_candidate_set())
        candidate_cams = scene.getCandidateCameras()
        
        scores = []
        for cam in tqdm(candidate_cams, desc="Uncertainty Evaluation"):
            if exit_func and exit_func():
                raise RuntimeError("Early exit requested")
            
            # Render uncertainty using Gaussian primitive data
            render_pkg = modified_render(
                cam, gaussians, pipe, background,
                override_color=gaussians.get_uncertainty()
            )
            
            # Get uncertainty map and calculate score
            uncertainty_map = render_pkg['uncertainty']
            scores.append(self.compute_uncertainty_score(uncertainty_map))

        # Select top N uncertain views
        _, indices = torch.topk(torch.tensor(scores), num_views)
        return [candidate_views[i] for i in indices.tolist()]
