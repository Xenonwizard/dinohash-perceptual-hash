import torch
import numpy as np
import torch.nn as nn
import math

# monkey patching  :)
def my_interpolate_pos_encoding(self, x, w, h):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed
    pos_embed = self.pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    M = int(math.sqrt(N))
    assert N == M * M
    kwargs = {}
    if self.interpolate_offset:
        sx = float(w0 + self.interpolate_offset) / M
        sy = float(h0 + self.interpolate_offset) / M
        kwargs["scale_factor"] = (sx, sy)
    else:
        kwargs["size"] = (w0, h0)
        
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        mode="bicubic",
        antialias=False, # MAIN CHANGE MADE HERE TO EXPORT TO ONNX
        **kwargs,
    )
    assert (w0, h0) == patch_pos_embed.shape[-2:]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

model_name = "vitb14_reg"
path = "./dinov2_0.0001_500_17982_20.pth"

# load model
dinov2 = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_name}').eval()
dinov2.interpolate_pos_encoding = my_interpolate_pos_encoding.__get__(dinov2, dinov2.__class__)
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

batch_size = 42  # can be set to anything
example = torch.rand(batch_size, 3, 224, 224)

dynamic_axes = {
    'input': {0: 'batch_size'},
    'output': {0: 'batch_size'}
}

onnx_program = torch.onnx.export(
    model,
    example,
    dynamo=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=dynamic_axes,
    do_constant_folding=True
)

onnx_program.optimize()
onnx_program.save(f"dinov2_{model_name}_{components.shape[1]}bit_dynamic.onnx")