import torch
import torchvision
import numpy as np
import torch.nn as nn

model_name = "vits14_reg"
path = "./train_models/dinov2_0.0001_500.0_20000_10.pth"

# Load model
dinov2 = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_name}').eval()
dinov2.load_state_dict(torch.load(path, weights_only=True))

means = np.load(f'./hashes/dinov2_{model_name}_means.npy')
means_torch = torch.from_numpy(means).float()

components = np.load(f'./hashes/dinov2_{model_name}_PCA.npy').T
components_torch = torch.from_numpy(components).float()

# integrate linear component to mimic PCA
linear = nn.Linear(components_torch.shape[0], components_torch.shape[1])
linear.weight.data = nn.parameter.Parameter(components_torch.T)
linear.bias.data = nn.parameter.Parameter(-means_torch@components_torch)

model = nn.Sequential(
    dinov2,
    linear
)

example = torch.rand(1, 3, 224, 224)

# use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save(f"dinov2_{model_name}_{components.shape[1]}bit.pt")