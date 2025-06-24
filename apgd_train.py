import argparse
import os

from PIL import Image
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import copy

import numpy as np
from hashes.dinohash import DINOHash, preprocess
import torch
from apgd_attack import APGDAttack, criterion_loss
from utils import AverageMeter

torch.manual_seed(0)
np.random.seed(0)

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

class ImageDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return preprocess(Image.open(self.image_files[idx]))

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Perform neural collision attack.')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=200,
                    help='batch size for processing images')
parser.add_argument('--image_dir', dest='image_dir', type=str,
                    default='./diffusion_data', help='directory containing images')
parser.add_argument('--n_iter', dest='n_iter', type=int, default=20,
                    help='average number of iterations')
parser.add_argument('--n_iter_range', dest='n_iter_range', type=int, default=0,
                    help='maximum number of iterations')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=8/255,
                    help='maximum perturbation (L∞ norm bound)')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=1,
                    help='number of epochs')
parser.add_argument('--lr', dest='lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--warmup', dest='warmup', type=int, default=1_400,
                    help='number of warmup steps')
parser.add_argument('--steps', dest='steps', type=int, default=20_000,
                    help='number of steps')
parser.add_argument('--start_step', dest='start_step', type=int, default=0,
                    help='starting step')
parser.add_argument('--clean_weight', dest='clean_weight', type=float, default=500,
                    help='weight of clean loss')
parser.add_argument('--val_freq', dest='val_freq', type=int, default=500,
                    help='validation frequency')
parser.add_argument('--resume_path', dest='resume_path', type=str, default=None,
                    help='resume path')

args = parser.parse_args()

image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
image_files.sort()
image_files = image_files[:1_800_000]

clean_dinohash = DINOHash(model="vits14_reg", pca_dims=96, prod_mode=False,)
adversarial_dinohash = DINOHash(model="vits14_reg", pca_dims=96, prod_mode=False)

dataset = ImageDataset(image_files)

for param in clean_dinohash.dinov2.parameters():
    param.requires_grad = False
clean_dinohash.dinov2.eval()

complete_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

SPLIT_RATIO = 0.999
train_dataset, test_dataset = random_split(dataset, [int(SPLIT_RATIO*len(dataset)), len(dataset)-int(SPLIT_RATIO*len(dataset))])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=11)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=11)

apgd = APGDAttack(dinohash=adversarial_dinohash, eps=args.epsilon)

if args.resume_path is not None:
    adversarial_dinohash.load_model(args.resume_path)

optimizer = AdamW(adversarial_dinohash.dinov2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
scheduler = cosine_lr(optimizer, args.lr, args.warmup, args.steps)
step_total = args.start_step
epoch_total = 0

for param in adversarial_dinohash.dinov2.parameters():
    param.requires_grad = True

while step_total < args.steps:
    pbar = tqdm(train_loader)
    loss_meter = AverageMeter('Loss')
    accuracy_meter = AverageMeter('Accuracy')

    for images in pbar:
        logits = clean_dinohash.hash(images, differentiable=False, logits=True).float().cuda()

        scheduler(step_total)
        n_iter = np.random.randint(args.n_iter - args.n_iter_range,
                                   args.n_iter + args.n_iter_range + 1)
        eps = np.random.uniform(0.5, 1.5) * args.epsilon

        logits = logits.cuda()
        images = images.cuda()

        adv_images, _ = apgd.attack_single_run(images, logits, n_iter, log=False, eps=eps)

        # adv_images = images.cuda()

        adversarial_dinohash.dinov2.train()
        adv_hashes, adv_loss = criterion_loss(adv_images, logits, loss="target bce")

        adv_loss = adv_loss.mean()
        adv_loss.backward()

        clean_loss = 0
        if args.clean_weight > 0:
            clean_hashes, clean_loss = criterion_loss(images, logits, loss="target bce")
            clean_loss =  args.clean_weight * clean_loss.mean()
            clean_loss.backward()

        loss = adv_loss + clean_loss
        adv_loss = adv_loss.item()

        if args.clean_weight > 0:
            clean_loss = clean_loss.item() / args.clean_weight
        else:
            clean_loss = 0

        optimizer.step()
        optimizer.zero_grad()

        hashes = (logits >= 0).float()
        accuracy = (adv_hashes - hashes).abs().mean()

        loss_meter.update(loss.item(), len(images))
        accuracy_meter.update(accuracy.item(), len(images))

        pbar.set_description(f"attack: {accuracy * 100:.4f}, loss: {loss:.4f}, adv_loss: {adv_loss:.4f}, clean_loss: {clean_loss:.4f}")

        step_total += 1

        del hashes, logits, images, adv_images, adv_hashes

        if step_total % args.val_freq == 0:
            adversarial_dinohash.dinov2.eval()

            total_strength = 0
            total_accuracy = 0
            n_images = 0

            for images in test_loader:
                logits = clean_dinohash.hash(images, differentiable=False, logits=True).float().cuda()
                hashes = (logits >= 0).float()

                adv_images, _ = apgd.attack_single_run(images, logits, n_iter=args.n_iter *2, eps=args.epsilon)

                adv_hashes = adversarial_dinohash.hash(adv_images).float()
                accuracy = (adv_hashes - hashes).cpu().abs().mean().item()

                clean_hashes = adversarial_dinohash.hash(images).float()
                clean_accuracy = (clean_hashes - hashes).cpu().abs().mean().item()

                total_strength += accuracy * len(images)
                total_accuracy += clean_accuracy * len(images)
                n_images += len(images)

                del hashes, logits, adv_images, adv_hashes, images, clean_hashes

            print(f"validation attack strength: {total_strength / n_images * 100:.2f}%, clean error:  {total_accuracy / n_images * 100:.2f}%")

        if step_total >= args.steps:
            break
        
    print(f"step: {step_total}, loss: {loss_meter.avg:.4f}, accuracy: {accuracy_meter.avg:.4f}")
    torch.save(adversarial_dinohash.dinov2.state_dict(), f'./dinov2_{args.lr}_{args.clean_weight}_{step_total}_{args.n_iter}.pth')

    del loss_meter, accuracy_meter