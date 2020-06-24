import os
import random
import sys

import imageio
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize

sys.path.insert(0, os.path.abspath((os.path.dirname(__file__)) + '/../'))
from classification.utils.transforms import corrupt_transform
from ocr.utils.transforms import four_point_transform, pad_transform
# TODO calculate mean and std over crops
from segmentation.utils.dataset import MEAN, STD


class ClassificationDataset(Dataset):
    def __init__(self, df, root, proba=0.1, transforms=None):
        self.image_filenames = df['file'].values
        self.boxes = df['box'].values if 'box' in df.columns else None
        self.texts = df['text'].values if 'text' in df.columns else None
        self.root = root
        self.proba = proba
        self.transforms = transforms

    def __getitem__(self, idx):
        if self.boxes is not None:
            image = Image.open(
                os.path.join(self.root, self.image_filenames[idx])
            ).convert('RGB')
            image = np.asarray(image)
        else:
            image = imageio.imread(
                os.path.join(self.root, self.image_filenames[idx])
            )[..., :3]

        if self.texts is None:
            image = pad_transform(image=image)

        if self.boxes is not None:
            box = self.boxes[idx]
            image = four_point_transform(image, box)

        if self.transforms:
            image = self.transforms(image=image)

        if self.texts is not None:
            if random.random() < self.proba:
                image = corrupt_transform(image=image)
                target = 0
            else:
                target = 1

        image = to_tensor(image)
        image = normalize(image, MEAN, STD)

        if self.texts is not None:
            return {'image': image, 'target': target}

        else:
            return {'image': image}

    def __len__(self):
        return len(self.image_filenames)
