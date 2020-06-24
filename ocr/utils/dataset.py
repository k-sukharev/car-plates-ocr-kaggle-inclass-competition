import os
import random
import sys

import imageio
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchtext import data
from torchvision.transforms.functional import to_tensor, normalize

sys.path.insert(0, os.path.abspath((os.path.dirname(__file__)) + '/../'))
from ocr.utils.transforms import pad_transform, four_point_transform
# TODO calculate mean and std over crops
from segmentation.utils.dataset import MEAN, STD


ALPHABET = "0123456789ABCEHKMOPTXY"

char_field = data.Field(
    tokenize=list,
    init_token='<bos>',
    eos_token='<eos>',
    fix_length=11
)
char_field.build_vocab(ALPHABET)

mapping = {
    'А': 'A',
    'В': 'B',
    'С': 'C',
    'Е': 'E',
    'Н': 'H',
    'К': 'K',
    'М': 'M',
    'О': 'O',
    'Р': 'P',
    'Т': 'T',
    'Х': 'X',
    'У': 'Y',
}


class OCRDataset(Dataset):
    def __init__(self, df, root, alphabet, transforms=None):
        self.image_filenames = df['file'].values
        self.boxes = df['box'].values if 'box' in df.columns else None
        self.texts = df['text'].values if 'text' in df.columns else None
        self.root = root
        self.alphabet = alphabet
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

        image = to_tensor(image)
        image = normalize(image, MEAN, STD)

        if self.texts is not None:
            text = self.texts[idx]
            seq = char_field.process([text]).squeeze(1)
            seq_shifted = seq[:-1]
            seq_gt = seq[1:]

            return {
                'image': image,
                'seq_shifted': seq_shifted,
                'seq_gt': seq_gt
            }

        else:
            return {'image': image}

    def __len__(self):
        return len(self.image_filenames)


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        d = random.randint(0, len(self.datasets)-1)
        return self.datasets[d][i % len(self.datasets[d])]

    def __len__(self):
        return max(len(d) for d in self.datasets)


def is_valid_str(s, alphabet=ALPHABET):
    for ch in s:
        if ch not in alphabet:
            return False
    return True


def convert_to_eng(text, mapping=mapping):
    return ''.join([mapping.get(a, a) for a in text])


def try_load(x, hparams):
    try:
        imageio.imread(
            os.path.join(os.path.join(hparams.data_path, 'generated_60k'), x)
        )
        return True
    except ValueError:
        return False
    except SyntaxError:
        return False


def make_plates_df(df):
    """Create a dataframe with entries for each car plate number"""
    ocr_df = df.explode('nums').dropna()
    ocr_df['box'] = ocr_df['nums'].apply(lambda x: x['box'])

    if ocr_df['nums'].iloc[0].get('text', 0):
        ocr_df['text'] = ocr_df['nums'].apply(
            lambda x: x['text'] if is_valid_str(x['text'])
            else convert_to_eng(x['text'].upper())
        )

    ocr_df = ocr_df.drop(columns=['nums'])
    return ocr_df
