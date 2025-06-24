import argparse
import os
from os.path import isfile, join
import numpy as np

from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from hashes.dinohash import preprocess, dinohash, dinov2, load_model
from apgd_attack import APGDAttack

load_model("train_models/dinov2_0.0001_500.0_20000_10.pth")

class ImageDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(args.image_dir, self.image_files[idx])).convert("RGB")
        return preprocess(image), self.image_files[idx]

parser = argparse.ArgumentParser(
    description='Perform neural collision attack.')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
                    help='batch size for processing images')
parser.add_argument('--image_dir', dest='image_dir', type=str,
                    default='./diffusion_data', help='directory containing images')
parser.add_argument('--n_iter', dest='n_iter', type=int, default=30,
                    help='number of iterations')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=8/255,
                    help='maximum perturbation (L∞ norm bound)')

args = parser.parse_args()

image_files = [f for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
image_files.sort()
image_files = image_files[-200_000:]

dataset = ImageDataset(image_files)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

apgd = APGDAttack(eps=args.epsilon)
dinov2.eval()
for param in dinov2.parameters():
    param.requires_grad = False

accs = 0
count = 0
for image_tensors, names in tqdm(dataloader):
    logits = dinohash(image_tensors, differentiable=False, logits=True, prod_output=False)
    adv_images, _ = apgd.attack_single_run(image_tensors, logits, 100)
    adv_logits = dinohash(adv_images, differentiable=False, logits=True, prod_output=False)
    acc = ((adv_logits>=0) == (logits>=0)).float().mean(1).cpu().numpy()
    accs += acc.sum().item()
    count += len(acc)
    print(f"Processed {count} images, current accuracy: {accs / count:.6f}")