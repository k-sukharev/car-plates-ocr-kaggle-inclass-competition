import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

from argparse import ArgumentParser

from imgaug.augmentables.heatmaps import HeatmapsOnImage
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath((os.path.dirname(__file__)) + '/../'))
from classification.run_training import CarPlatesClassifier
from classification.utils.dataset import ClassificationDataset
from ocr.run_training import TransformerOCR
from ocr.utils.dataset import ALPHABET, make_plates_df, char_field, OCRDataset
from ocr.utils.transforms import ocr_transforms, pad_transform
from segmentation.run_training import FPN
from segmentation.utils.dataset import SegmentationDataset
from segmentation.utils.transforms import configure_transforms

MIN_AREA = 250
SUBMISSION_PATH = 'sub_transformer_ocr.csv'


def decode(seq):
    seq = seq.permute(1, 0).tolist()
    result = []
    for s in seq:
        string = ''
        for i in s:
            if (
                i == char_field.vocab.stoi[char_field.init_token] or
                i == char_field.vocab.stoi[char_field.pad_token]
            ):
                continue
            elif i == char_field.vocab.stoi[char_field.eos_token]:
                break
            else:
                string += char_field.vocab.itos[i]
        result.append(string)
    return result


def make_segmentation_preds(test_df, hparams, device):
    model = FPN.load_from_checkpoint(
        hparams.segmentation_checkpoint_path
    )
    model.freeze()
    model.to(device)

    _, valid_transforms = configure_transforms(
        model.hparams.resize_to
    )
    test_dataset = SegmentationDataset(
        df=test_df,
        root=hparams.data_path,
        transforms=valid_transforms
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=model.hparams.batch_size,
        num_workers=model.hparams.num_workers
    )

    segmentation_predictions = np.zeros((
        len(test_dataset),
        1,
        model.hparams.resize_to,
        model.hparams.resize_to
    ))

    for i, batch in enumerate(tqdm(
            test_dataloader,
            total=len(test_dataloader),
            desc='segmentation test prediction...'
    )):
        image = batch['image'].to(device)
        mask_logits = model(image)
        mask_pr = torch.sigmoid(mask_logits).cpu().numpy()
        segmentation_predictions[
            i * test_dataloader.batch_size : (i + 1) * test_dataloader.batch_size
        ] = mask_pr

    return segmentation_predictions


def process_segmentation_preds(segmentation_predictions, test_df):
    nums = [None for i in range(len(test_df))]

    for i in tqdm(range(len(test_df)), desc='postprocessing...'):
        image = Image.open(os.path.join(hparams.data_path, test_df['file'][i])).convert('RGB')
        image = np.asarray(image)
        image = pad_transform(image=image)

        mask = segmentation_predictions[i][0].astype(np.float32)
        heatmap = HeatmapsOnImage(mask, shape=image.shape[:2])
        heatmap = heatmap.resize(image.shape[:2])
        mask = heatmap.get_arr()
        mask = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [
            np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
            for contour in contours if cv2.contourArea(contour) > MIN_AREA
        ]
        nums[i] = [{'box': contour.tolist()} for contour in contours]
    return nums


def make_classification_preds(plates_test_df, hparams, device):
    test_dataset = ClassificationDataset(
        df=plates_test_df,
        root=hparams.data_path,
        transforms=ocr_transforms
    )

    model = CarPlatesClassifier.load_from_checkpoint(
        hparams.classification_checkpoint_path
    )
    model.freeze()
    model.to(device)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=model.hparams.batch_size,
        num_workers=model.hparams.num_workers
    )

    classification_predictions = []

    for batch in tqdm(test_loader, desc='classification test prediction...'):
        image = batch['image'].to(device)
        logits = model(image)
        classification_predictions.append(logits.cpu())

    classification_predictions = torch.sigmoid(
        torch.cat(classification_predictions)
    ).numpy().flatten()

    return classification_predictions


def make_ocr_preds(plates_test_df, hparams, device):
    test_dataset = OCRDataset(
        df=plates_test_df,
        root=hparams.data_path,
        alphabet=ALPHABET,
        transforms=ocr_transforms
    )

    model = TransformerOCR.load_from_checkpoint(
        hparams.ocr_checkpoint_path
    )
    model.freeze()
    model.to(device)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=model.hparams.batch_size,
        num_workers=model.hparams.num_workers
    )

    ocr_predictions = []

    for batch in tqdm(test_loader, desc='ocr test prediction...'):
        image = batch['image'].to(device)
        bs = len(image)
        seq = torch.LongTensor(
            np.ones((1, bs)) * char_field.vocab[char_field.init_token]
        ).to(device)
        for _ in range(1, char_field.fix_length):
            tgt_mask = model.transf.generate_square_subsequent_mask(len(seq)).to(device)
            seqs_pr = model(image, seq, tgt_mask=tgt_mask)
            seq = torch.cat([seq, seqs_pr[-1:].argmax(dim=-1)])
        ocr_predictions.append(seq)

    ocr_predictions = torch.cat(ocr_predictions, dim=1).cpu()

    return ocr_predictions


def create_submission(test_df, test_plates_idxs_to_erase, pred_nums):
    sub = make_plates_df(test_df)
    sub['plates_string'] = pred_nums
    sub.iloc[test_plates_idxs_to_erase, 2] = ''
    sub = sub.sort_values(['box'])
    # remove plates shorter than 7 symbols
    sub['plates_string'][sub['plates_string'].str.len() <= 6] = ''
    sub = sub[['file', 'plates_string']].groupby('file').agg(' '.join) \
        .reset_index().rename(columns={'file': 'file_name'})
    # create entries for files with no plates detected
    exploded = test_df.explode('nums')
    missing = exploded[exploded['nums'].isna()] \
        .rename(columns={'file': 'file_name', 'nums': 'plates_string'}).fillna('')
    sub = pd.concat([sub, missing], ignore_index=True)
    # hack to remove unwanted spaces
    sub['plates_string'] = sub['plates_string'].apply(
        lambda x: ' '.join(x.split())
    )
    return sub


def main(hparams):
    device = f'cuda:{hparams.gpu}' if torch.cuda.is_available() else 'cpu'

    test_df = pd.DataFrame(
        {'file': 'test/' + pd.Series(sorted(
            os.listdir(os.path.join(hparams.data_path, 'test')),
            key=lambda x: int(x.split('.')[0])
        ))}
    )

    # segmentation inference
    segmentation_predictions = make_segmentation_preds(test_df, hparams, device)
    test_df['nums'] = process_segmentation_preds(segmentation_predictions, test_df)
    plates_test_df = make_plates_df(test_df)

    # classification inference
    classification_predictions = make_classification_preds(plates_test_df, hparams, device)
    test_plates_idxs_to_erase = np.arange(len(plates_test_df))[classification_predictions < 0.5]

    # ocr inference
    ocr_predictions = make_ocr_preds(plates_test_df, hparams, device)
    pred_nums = decode(ocr_predictions)

    # creating submission
    sub = create_submission(test_df, test_plates_idxs_to_erase, pred_nums)
    sub.to_csv(SUBMISSION_PATH, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="select GPU device")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/xfs/Datasets/data/",
        help="path where dataset is stored"
    )

    parser.add_argument(
        "--segmentation_checkpoint_path",
        type=str,
        help="path to segmentation model checkpoint"
    )

    parser.add_argument(
        "--classification_checkpoint_path",
        type=str,
        help="path to classification model checkpoint"
    )

    parser.add_argument(
        "--ocr_checkpoint_path",
        type=str,
        help="path to ocr model checkpoint"
    )

    hparams = parser.parse_args()

    main(hparams)
