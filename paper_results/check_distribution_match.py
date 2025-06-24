import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from sklearn.covariance import empirical_covariance
import seaborn as sns

from hashes.dinohash import dinohash, preprocess, load_model

class ImageDataset(Dataset):
    def __init__(self):
        file_name = "test_img_filepath.txt"
        with open(file_name, "r") as f:
            lines = f.readlines()
        self.image_files = [line.strip() for line in lines]
        print("Total images detected:", len(self.image_files))
        self.image_files = ["../SMP_test_images/" + f[5:] for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        return preprocess(image)

hasher = dinohash
load_model("train_models/dinov2_0.0001_500.0_20000_10.pth")
BATCH_SIZE = 256

hashes = []
dataset = ImageDataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=11)

for images in tqdm(dataloader):
    hashes_batch = hasher(images, prod_output=False).tolist()
    hashes.extend(hashes_batch)

hashes = np.array(hashes)
covariance = empirical_covariance(hashes)
mixed_variance = np.diag(covariance).T[:, np.newaxis] @ np.diag(covariance)[np.newaxis, :]
covariance /= np.sqrt(mixed_variance)
means = np.mean(hashes, axis=0)

heatmap = sns.heatmap(np.abs(covariance), cmap="coolwarm", cbar=True, square=True)
fig = heatmap.get_figure()
fig.savefig("covariance.png")
print("Mean:", means)