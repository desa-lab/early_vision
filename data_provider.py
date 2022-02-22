"""Helper function for data provider."""
import glob
import os

import numpy as np
import torch  # pylint:disable=import-error
from PIL import Image

import torchvision.transforms as transforms  # pylint:disable=import-error
import torchvision.transforms.functional as F  # pylint:disable=import-error
import torch.nn.functional as nnf


def get_train_val_ids():
    """Get image ids for train and val."""
    train_ids = glob.glob(
        "/home/vveeraba/src/v1net_bsds/data/BSDS500/data/images/train/*jpg")
    val_ids = glob.glob(
        "/home/vveeraba/src/v1net_bsds/data/BSDS500/data/images/val/*jpg")
    train_ids = [i.split("/")[-1].split(".")[0] for i in train_ids]
    val_ids = [i.split("/")[-1].split(".")[0] for i in val_ids]
    return {"train": train_ids, "val": val_ids}


class BSDSDataProvider():
    """Data provider for BSDS500."""

    def __init__(self,
                 crop_size,
                 is_training,
                 data_dir,
                 noise_sigma,
                 transform,
                 downsample=4,
                 ):
        self.crop_size = crop_size
        self.is_training = is_training
        self.data_dir = data_dir
        self.img_gt_paths = []
        self.noise_sigma = noise_sigma
        self.img_transform = transform
        self.downsample = downsample

        if self.is_training:
            self.data_file = os.path.join(self.data_dir, "train_pair.lst")
            self.img_ids = get_train_val_ids()["train"]
        else:
            self.data_file = os.path.join(self.data_dir, "train_pair.lst")
            self.img_ids = get_train_val_ids()["val"]

        with open(self.data_file, "r") as f:
            all_files = f.read()

        all_files = all_files.strip().split("\n")
        for f in all_files:
            img, gt = f.split(" ")
            img_id = img.split("/")[-1].split(".")[0]
            if img_id in self.img_ids:
                self.img_gt_paths.append((img, gt))
        self.num_samples = len(self.img_gt_paths)

    def noise_transform(self, imgs):
        """Generate input, label pairs for image denoising/superresolution."""
        if self.img_transform.startswith('gaussian'):
            noise_sigma = self.noise_sigma/255.
            noisy_img = torch.clip(imgs.detach() +
                                   torch.normal(0, noise_sigma,
                                                imgs.size()),
                                   0., 1.)
            return imgs, noisy_img
        elif self.img_transform.startswith('superres'):
            small_imgs = F.resize(
                imgs, (imgs.size(1)//self.downsample, imgs.size(2)//self.downsample))
            resized_imgs = F.resize(small_imgs, (imgs.size(1), imgs.size(2)))
            return imgs, resized_imgs
        raise NotImplementedError("transform not implemented yet")

    def transform(self, images, xmax=1.):
        """Transform images and ground truth."""
        images = np.array(images)
        if images.max() > 1.:
            images = images / 255
        images = np.float32(images)
        images = F.to_tensor(images * xmax)
        i, j, h, w = transforms.RandomCrop.get_params(
            images, output_size=(self.crop_size, self.crop_size))
        images_cropped = F.crop(images, i, j, h, w)
        _, transformed_imgs = self.noise_transform(images_cropped)
        return images_cropped, transformed_imgs

    def __getitem__(self, idx):
        img, _ = self.img_gt_paths[idx]
        img = os.path.join(self.data_dir, img)
        img = Image.open(img).convert("L")
        img, noisy_img = self.transform(img)
        return noisy_img, img

    def __len__(self):
        return self.num_samples
