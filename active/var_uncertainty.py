import torch
from gaussian_renderer import forward_k_times

class VarUncertaintySelector:
    def __init__(self, args):
        self.args = args

    def nbvs(self, gaussians, scene, num_views, pipe, background, exit_func=None):
        candidate_views = scene.getTrainCameras()  # All possible views
        if scene.candidate_views_filter:
            candidate_views = [cam for i, cam in enumerate(candidate_views) if i in scene.candidate_views_filter]

        # Compute uncertainty for each candidate view
        uncertainties = []
        for viewpoint in candidate_views:
            if exit_func and exit_func():
                raise RuntimeError("Early exit triggered by cluster manager")
            
            # Render with variational uncertainty
            render_pkg = forward_k_times(viewpoint, gaussians, pipe, background, k=gaussians.n_models)
            uncertainty = render_pkg['comp_std'].mean()  # Mean standard deviation as uncertainty metric
            uncertainties.append(uncertainty.item())

        # Sort views by uncertainty and select top num_views
        sorted_indices = torch.argsort(torch.tensor(uncertainties), descending=True)
        selected_views = [candidate_views[i].uid for i in sorted_indices[:num_views]]

        return selected_views