import os
import sys

import pandas as pd
import pretrainedmodels
import pytorch_lightning as pl

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
from classification.utils.dataset import ClassificationDataset
from ocr.utils.dataset import make_plates_df
from ocr.utils.transforms import ocr_transforms
from segmentation.run_training import FILES_IDXS_TO_DROP

CHECKPOINT_DIR = './checkpoints/classification/'


class CarPlatesClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.net = pretrainedmodels.resnet50()
        self.net.last_linear = nn.Linear(self.net.last_linear.in_features, 1)

        self.loss = L.SoftBCEWithLogitsLoss()

    def forward(self, x):
        logits = self.net(x)
        return logits

    def training_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target'].unsqueeze(1).float()

        logits = self.forward(image)
        loss = self.loss(logits, target)

        acc = ((torch.sigmoid(logits) > 0.5) == target).float().mean()

        log_dict = {'train_loss': loss, 'acc': acc}

        return {'loss': loss, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target'].unsqueeze(1).float()

        logits = self.forward(image)
        loss = self.loss(logits, target)

        val_acc = ((torch.sigmoid(logits) > 0.5) == target).float().mean()

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        log_dict = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}

        return {
            'val_loss': avg_loss,
            'log': log_dict,
            'progress_bar': log_dict
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr
        )

        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.hparams.scheduler_gamma
        )

        return [optimizer], [scheduler]

    def prepare_data(self):
        train_annot = pd.read_json(
            os.path.join(hparams.data_path, 'train.json')
        ).drop(index=FILES_IDXS_TO_DROP)

        train_df, valid_df = train_test_split(
            train_annot,
            test_size=hparams.valid_size,
            shuffle=True,
            random_state=hparams.seed
        )

        plates_train_df = make_plates_df(train_df)
        plates_valid_df = make_plates_df(valid_df)

        self.train_dataset = ClassificationDataset(
            df=plates_train_df,
            root=hparams.data_path,
            transforms=ocr_transforms,
        )

        self.valid_dataset = ClassificationDataset(
            df=plates_valid_df,
            root=hparams.data_path,
            transforms=ocr_transforms,
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

    model = CarPlatesClassifier(hparams)

    checkpoint_callback = ModelCheckpoint(
        filepath=CHECKPOINT_DIR,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=1,
        gpus=[hparams.gpu] if torch.cuda.is_available() else None,
        max_epochs=hparams.epochs,
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
        "--batch_size",
        type=int,
        default=128,
        help="size of the batches"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="number of workers for dataloader"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
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
