# DINOHash Perceptual Hash

This repository provides functionality for generating perceptual hashes for images using DinoHash

NOTE: we suggest using `git clone ... --depth 1` while setting up the repository since there were large files in previous commits that might cause git to malfunction

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