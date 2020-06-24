import os
import sys

import pandas as pd
import pytorch_lightning as pl
import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_toolbelt import losses as L
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath((os.path.dirname(__file__)) + '/../'))
from segmentation.run_training import FILES_IDXS_TO_DROP
from ocr.utils.dataset import ALPHABET, ConcatDataset, convert_to_eng, \
    char_field, make_plates_df, OCRDataset, try_load
from ocr.utils.transforms import ocr_synthetic_transforms, ocr_transforms, \
    pad_transform

CHECKPOINT_DIR = './checkpoints/ocr/'


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = nn.Parameter(torch.randn(1, max_len, d_model))
        self.register_parameter('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerOCR(pl.LightningModule):
    def __init__(self, hparams, vocab_size=len(char_field.vocab)):
        super().__init__()
        self.hparams = hparams

        self.feature_extractor = nn.Sequential(*list(
            pretrainedmodels.se_resnext50_32x4d().children()
        )[:-3]) # feature extractor not including last se block and fc layers

        self.ext = list(
            pretrainedmodels.se_resnext50_32x4d().children()
        )[-3]  # last se block of feature extractor

        self.fc = nn.Linear(
            self.feature_extractor[-1][-1].se_module.fc2.out_channels,
            hparams.d_model
        )

        self.fc_ext = nn.Linear(
            self.ext[-1].se_module.fc2.out_channels,
            hparams.d_model
        )

        self.pe1 = PositionalEncoding(hparams.d_model, hparams.dropout)

        self.pe1_ext = PositionalEncoding(hparams.d_model, hparams.dropout)

        self.emb = nn.Embedding(vocab_size, hparams.d_model)
        self.pe2 = PositionalEncoding(hparams.d_model, hparams.dropout)

        self.transf = nn.Transformer(
            hparams.d_model,
            hparams.nhead,
            num_encoder_layers=hparams.num_encoder_layers,
            num_decoder_layers=hparams.num_decoder_layers,
            dim_feedforward=hparams.dim_ff
        )

        self.transf_ext = nn.Transformer(
            hparams.d_model,
            hparams.nhead,
            num_encoder_layers=hparams.num_encoder_layers,
            num_decoder_layers=hparams.num_decoder_layers,
            dim_feedforward=hparams.dim_ff
        )

        self.fc_out = nn.Linear(2 * hparams.d_model, vocab_size)

        self.loss = L.SoftCrossEntropyLoss(ignore_index=char_field.vocab.stoi[char_field.pad_token])

        for p in self.transf.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.transf_ext.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, tgt_mask=None):
        src = self.feature_extractor(src)
        src_ext = self.ext(src)

        src = src.flatten(2).permute(2, 0, 1)
        src = self.pe1(self.fc(src))

        src_ext = src_ext.flatten(2).permute(2, 0, 1)
        src_ext = self.pe1_ext(self.fc_ext(src_ext))

        tgt = self.emb(tgt)
        tgt = self.pe2(tgt)

        x = self.transf(src, tgt, tgt_mask=tgt_mask)

        x_ext = self.transf_ext(src_ext, tgt, tgt_mask=tgt_mask)

        logits = self.fc_out(torch.cat([x, x_ext], dim=-1))

        return logits

    def training_step(self, batch, batch_idx):
        image = batch['image']
        seq_shifted = batch['seq_shifted'].permute(1, 0)
        seq_gt = batch['seq_gt'].permute(1, 0)
        tgt_mask = self.transf.generate_square_subsequent_mask(
            len(seq_shifted)
        ).type_as(image)

        seq_pr = self.forward(image, seq_shifted, tgt_mask=tgt_mask)

        loss = self.loss(seq_pr.reshape(-1, len(char_field.vocab)), seq_gt.flatten())

        acc = (seq_pr.argmax(-1) == seq_gt).float().mean()

        log_dict = {'train_loss': loss, 'acc': acc}

        return {'loss': loss, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        seq_shifted = batch['seq_shifted'].permute(1, 0)
        seq_gt = batch['seq_gt'].permute(1, 0)
        tgt_mask = self.transf.generate_square_subsequent_mask(
            len(seq_shifted)
        ).type_as(image)

        seq_pr = self.forward(image, seq_shifted, tgt_mask=tgt_mask)
        loss = self.loss(seq_pr.reshape(-1, len(char_field.vocab)), seq_gt.flatten())

        val_acc = (seq_pr.argmax(-1) == seq_gt).float().mean()

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

        filenames = pd.Series(
            os.listdir(os.path.join(hparams.data_path, 'generated_60k'))
        )
        error_files = filenames.apply(lambda x: try_load(x, hparams))
        text = filenames.apply(lambda x: convert_to_eng(x.split('.')[0].upper()))

        plates_synthetic_train_df = pd.DataFrame(
            {
                'file': 'generated_60k/' + filenames,
                'text': text

            }
        )[error_files]

        train_dataset = OCRDataset(
            df=plates_train_df,
            root=self.hparams.data_path,
            alphabet=ALPHABET,
            transforms=ocr_synthetic_transforms,
        )

        synthetic_train_dataset = OCRDataset(
            df=plates_synthetic_train_df,
            root=self.hparams.data_path,
            alphabet=ALPHABET,
            transforms=ocr_synthetic_transforms,
        )

        self.train_dataset = ConcatDataset(
            train_dataset,
            synthetic_train_dataset
        )

        self.valid_dataset = OCRDataset(
            df=plates_valid_df,
            root=self.hparams.data_path,
            alphabet=ALPHABET,
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

    model = TransformerOCR(hparams)

    checkpoint_callback = ModelCheckpoint(
        filepath=CHECKPOINT_DIR,
        verbose=True,
        monitor='avg_val_acc',
        mode='max'
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
        "--d_model",
        type=int,
        default=256,
        help="number of expected features in the transformer encoder/decoder inputs"
    )
    parser.add_argument(
        "--dim_ff",
        type=int,
        default=256,
        help="dimension of the feedforward network model in transformer"
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="number of heads in the multi head attention"
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=2,
        help="number of transformer encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=2,
        help="number of transformer decoder layers"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
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
