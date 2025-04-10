"""
This module provides functionality for loading a pre-trained DINOv2 model and generating perceptual hashes for images.

Classes:
    Hash:
        A class to handle the perceptual hash tensor and provide various conversion methods.
        Methods:
            __init__(tensor: torch.Tensor):
                Initializes the Hash object with a given tensor.
            to_hex() -> str:
                Converts the hash tensor to a hexadecimal string.
            to_string() -> str:
                Converts the hash tensor to a binary string.
            to_pytorch() -> torch.Tensor:
                Returns the hash tensor as a PyTorch tensor.
            to_numpy() -> np.ndarray:
                Returns the hash tensor as a NumPy array.

Functions:
    load_model(path: str) -> None:
        Loads the DINOv2 model state from the specified path.
    dinohash(image_arrays: Union[np.ndarray, List[Image.Image]]) -> torch.Tensor:
        Generates perceptual hashes for the given images using the DINOv2 model.
        Parameters:
            image_arrays (Union[np.ndarray, List[Image.Image], torch.Tensor]): Input images as a numpy array or a list of PIL Images or a torch.Tensor.
            differentiable (bool): If True, enables gradient computation. Default is False.
            mydinov2 (torch.nn.Module): The DINOv2 model to use for generating hashes. Default is the globally loaded model.
        Returns:
            torch.Tensor: The generated perceptual hashes.
"""

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from typing import Union, List
import sys


class Hash:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor.cpu()
        self.string = ''.join(str(int(x)) for x in self.tensor)
        self.hex = hex(int(self.string, 2))
        self.array = self.tensor.numpy()


def load_model(path):
    global dinov2
    dinov2.load_state_dict(torch.load(path, weights_only=True))

model = "vitb14_reg"
# Load model
dinov2 = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model}').cuda()
for param in dinov2.parameters():
    param.requires_grad = False
dinov2.eval()

means = np.load(f'./hashes/dinov2_{model}_means.npy')
means_torch = torch.from_numpy(means).cuda().float()

components = np.load(f'./hashes/dinov2_{model}_PCA.npy').T
components_torch = torch.from_numpy(components).cuda().float()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def dinohash(
    image_arrays: Union[np.ndarray, List[Image.Image], torch.Tensor],
    differentiable: bool = False,
    c: int = 1,
    logits: bool = False,
    l2_normalize: bool = False,
    mydinov2: torch.nn.Module = dinov2,
    prod_output: bool = True
    ) -> torch.Tensor:

    wrapper = torch.no_grad if not differentiable else torch.enable_grad

    if isinstance(image_arrays, np.ndarray):
        image_arrays = torch.from_numpy(image_arrays)
    if isinstance(image_arrays[0], Image.Image):
        image_arrays = torch.stack([preprocess(im) for im in image_arrays])
    if isinstance(image_arrays[0], str):
        image_arrays = torch.stack([preprocess(Image.open(im)) for im in image_arrays])

    with wrapper():
        image_arrays = normalize(image_arrays.cuda())
        
        outs = mydinov2(image_arrays) - means_torch
        
        outs = outs @ components_torch

        if l2_normalize:
            outs = torch.nn.functional.normalize(outs, dim=1)
        outs *= c

        if not logits:
            if differentiable:
                outs = torch.sigmoid(outs)
            else:
                outs = outs >= 0

    del image_arrays

    if prod_output:
        return [Hash(out) for out in outs]
    return outs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dinohash.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = Image.open(image_path)
    hash_tensor = dinohash([image])[0].hex
    print("Perceptual hash:", hash_tensor)

