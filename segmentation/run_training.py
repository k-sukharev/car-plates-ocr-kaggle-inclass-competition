import os
import sys

import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn

from argparse import ArgumentParser

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_toolbelt import losses as L
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath((os.path.dirname(__file__)) + '/../'))
from segmentation.utils.dataset import SegmentationDataset
from segmentation.utils.transforms import configure_transforms


FILES_IDXS_TO_DROP = [114, 249, 290, 413, 633, 712, 869, 929, 937, 944, 972, 1010, 1066, 1123, 1234, 1256, 1272, 1393, 1485, 1507, 1828, 1937, 2634, 2863, 3063, 3087, 3122, 3250, 3260, 3377, 3531, 3583, 3587, 3881, 3890, 3964, 4190, 4364, 4386, 4441, 4502, 4549, 4597, 4660, 4820, 5066, 5152, 5167, 5174, 5308, 5418, 5425, 5432, 5488, 5683, 5706, 5759, 5850, 5954, 6050, 6078, 6235, 6599, 6732, 6912, 7147, 7478, 7958, 8081, 8406, 8410, 8612, 8631, 8684, 8889, 9242, 9478, 9519, 9672, 10086, 10223, 10568, 10622, 10777, 10981, 11178, 11218, 11248, 11317, 11324, 11355, 11365, 11456, 11639, 11787, 11820, 12085, 12165, 12270, 12349, 12385, 12400, 12599, 12634, 12826, 13304, 13758, 13808, 13814, 13886, 13898, 13925, 14189, 14300, 14790, 14862, 14811, 14923, 14934, 15103, 15340, 15459, 15517, 15727, 15781, 15961, 16115, 16276, 16292, 16306, 16493, 16621, 16718, 16722, 16938, 17051, 17166, 17240, 17250, 17337, 17450, 17776, 17789, 17872, 17911, 17962, 18128, 18179, 18314, 18352, 18517, 18598, 18786, 18933, 19053, 19066, 19242, 19293, 19626, 19694, 19726, 19741, 19803, 19890, 20019, 20119, 20191, 20270, 20509, 20553, 20781, 20864, 21019, 21173, 21183, 21309, 21323, 21349, 21369, 21484, 21503, 21570, 22027, 22140, 22286, 23229, 22391, 22393, 22546, 22687, 22810, 22822, 22876, 22902, 22972, 23089, 23218, 23514, 23558, 23665, 23624, 23701, 23758, 23778, 23788, 23877, 23955, 24117, 24124, 24178, 24061, 24456, 24494, 24526, 24536, 24583, 24713, 24832, 24868, 25008, 25047, 25057, 25064, 25201, 25282, 25287, 25294, 25334, 25403, 25558, 25632]

CHECKPOINT_DIR = './checkpoints/segmentation/'


class FPN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.fpn = smp.FPN(
            encoder_name=hparams.encoder_name
        )

        self.iou = smp.utils.metrics.IoU(activation='sigmoid')
        self.mixed_loss = L.JointLoss(
            L.BinaryFocalLoss(),
            L.BinaryLovaszLoss(),
            0.7,
            0.3
        )

    def forward(self, x):
        x = self.fpn(x)
        return x

    def training_step(self, batch, batch_idx):
        image = batch['image']
        mask_gt = batch['mask']

        mask_logits = self.forward(image)

        loss = self.mixed_loss(mask_logits, mask_gt)
        iou = self.iou(mask_logits, mask_gt)

        log_dict = {'train_loss': loss, 'iou': iou}

        return {'loss': loss, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        mask_gt = batch['mask']

        mask_logits = self.forward(image)

        loss = self.mixed_loss(mask_logits, mask_gt)
        val_iou = self.iou(mask_logits, mask_gt)

        return {'val_loss': loss, 'val_iou': val_iou}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_iou = torch.stack([x['val_iou'] for x in outputs]).mean()

        log_dict = {'val_loss': avg_loss, 'avg_val_iou': avg_val_iou}

        return {
            'val_loss': avg_loss,
            'log': log_dict,
            'progress_bar': log_dict
        }

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {
                'params': self.fpn.encoder.parameters(),
                'lr': self.hparams.encoder_lr,
                'weight_decay': self.hparams.encoder_weight_decay
            },
            {'params': self.fpn.decoder.parameters()},
            {'params': self.fpn.segmentation_head.parameters()},
        ], lr=self.hparams.lr)

        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.hparams.scheduler_gamma
        )

        return [optimizer], [scheduler]

    def prepare_data(self):
        train_annot = pd.read_json(
            os.path.join(self.hparams.data_path, 'train.json')
        ).drop(index=FILES_IDXS_TO_DROP)

        train_df, valid_df = train_test_split(
            train_annot,
            test_size=self.hparams.valid_size,
            shuffle=True,
            random_state=self.hparams.seed
        )

        train_transforms, valid_transforms = configure_transforms(
            self.hparams.resize_to
        )

        self.train_dataset = SegmentationDataset(
            df=train_df,
            root=self.hparams.data_path,
            transforms=train_transforms,
        )

        self.valid_dataset = SegmentationDataset(
            df=valid_df,
            root=self.hparams.data_path,
            transforms=valid_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )


def main(hparams):
    seed_everything(hparams.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(hparams.gpu)

    model = FPN(hparams)

    checkpoint_callback = ModelCheckpoint(
        filepath=CHECKPOINT_DIR,
        verbose=True,
        monitor='avg_val_iou',
        mode='max'
    )

    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=1,
        gpus=[hparams.gpu] if torch.cuda.is_available() else None,
        max_epochs=hparams.epochs,
        amp_level='O1',
        precision=16,
        deterministic=True
    )

    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="select GPU device")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/xfs/Datasets/data/",
        help="path where dataset is stored"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--valid_size",
        type=float,
        default=0.1,
        help="validation size"
    )
    parser.add_argument(
        "--resize_to",
        type=int,
        default=512,
        help="size of the batches"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="size of the batches"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="number of workers for dataloader"
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="se_resnext50_32x4d",
        help="FPN encoder"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train"
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=0.0005,
        help="encoder learning rate"
    )
    parser.add_argument(
        "--encoder_weight_decay",
        type=float,
        default=0.00003,
        help="encoder weight decay"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate"
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.8,
        help="gamma for ExponentialLR scheduler"
    )

    hparams = parser.parse_args()

    main(hparams)
