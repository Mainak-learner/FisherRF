# create_config_for_viewer.py

import yaml
import os

# Path where your visualization folder is
vis_folder = "my_vis"

# Config contents
config = {
    "trainer": {
        "steps_per_eval_batch": 500,
        "steps_per_save": 1000,
        "steps_per_eval_image": 1000,
        "max_num_iterations": 10000,
        "mixed_precision": True,
    },
    "method_name": "vanilla-nerf",
    "pipeline": {
        "datamanager": {
            "data": "./transforms.json",  # Nerfstudio will find your transforms.json here
            "camera_type": "spherical",   # best match if you're generating poses around object
        },
        "model": {
            "_target": "nerfstudio.models.vanilla_nerf.VanillaModelConfig",
            "eval_num_rays_per_chunk": 4096,
        },
    },
    "dataset": {
        "batch_size": 4096,
    },
    "vis": {
        "viewer_activated": True,
    }
}

# Save it
os.makedirs(vis_folder, exist_ok=True)
with open(os.path.join(vis_folder, "config.yml"), "w") as f:
    yaml.dump(config, f, sort_keys=False)

print(f"[Done] Created {vis_folder}/config.yml successfully!")
