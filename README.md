[![DOI](https://zenodo.org/badge/822733254.svg)](https://doi.org/10.5281/zenodo.15525403) [![arXiv](https://img.shields.io/badge/arXiv-10.48550/arXiv.2503.11195-b31b1b.svg)](https://doi.org/10.48550/arXiv.2503.11195) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/proteus-photos/dinohash-perceptual-hash)

# DINOHash Perceptual Hash

This repository provides functionality for generating perceptual hashes for images using DinoHash

NOTE: we suggest using `git clone ... --depth 1` while setting up the repository since there were large files in previous commits that might cause git to malfunction

## Node.js Implementation
To use the Node.js package look at the README in the `node_package` folder

## Setup
Install the latest version of PyTorch according to your OS. Then run
```
pip install transformers numpy pillow
```

## Usage

### dinohash Function

The `dinohash` function generates perceptual hashes for the given images using the DINOv2 model.

#### Parameters

- `image_arrays` (Union[np.ndarray, List[Image.Image], torch.Tensor]): Input images as a numpy array, a list of PIL Images, or a torch.Tensor.

#### Returns

- `List[Hash]`: The generated perceptual hashes.

#### Example

```python
from PIL import Image
import torch
from dinohash import dinohash

# Load an image
image_path = "path/to/your/image.jpg"
image = Image.open(image_path)

# Generate perceptual hash
hashes = dinohash([image])

# Print the hexadecimal representation of the hash
print("Perceptual hash:", hashes[0].hex)
```

### Command Line Usage

You can also use the `dinohash.py` script from the command line to generate perceptual hashes for an image.

```sh
python dinohash.py <image_path>
```

Replace `<image_path>` with the path to your image file.

Example:

```sh
python dinohash.py path/to/your/image.jpg
```

This will print the perceptual hash of the image.
