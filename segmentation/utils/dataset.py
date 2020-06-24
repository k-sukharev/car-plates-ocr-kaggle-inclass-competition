import os

import numpy as np

from imgaug.augmentables.polys import Polygon
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize


MEAN = [0.4161, 0.4293, 0.4289]
STD = [0.2421, 0.2409, 0.2459]


class SegmentationDataset(Dataset):
    def __init__(self, df, root, transforms=None):
        self.image_filenames = df['file'].values
        self.nums = df['nums'].values if 'nums' in df.columns else None
        self.root = root
        self.transforms = transforms

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.root, self.image_filenames[idx])
        ).convert('RGB')

        image = np.asarray(image)

        if self.nums is not None:
            segmap = np.zeros(
                (image.shape[0], image.shape[1], 1),
                dtype=np.int32
            )

            nums = self.nums[idx]

            for num in nums:
                poly = Polygon(num['box'])
                segmap = poly.draw_on_image(segmap, color=2)

            segmap = SegmentationMapsOnImage(segmap, shape=image.shape)

            if self.transforms:
                image, segmap = self.transforms(
                    image=image,
                    segmentation_maps=segmap
                )

            image = to_tensor(image)
            image = normalize(image, MEAN, STD)

            segmap = segmap.arr
            segmap = to_tensor(segmap.astype(np.float32))

            return {'image': image, 'mask': segmap}

        else:
            if self.transforms:
                image = self.transforms(image=image)

            image = to_tensor(image)
            image = normalize(image, MEAN, STD)

            return {'image': image}

    def __len__(self):
        return len(self.image_filenames)
