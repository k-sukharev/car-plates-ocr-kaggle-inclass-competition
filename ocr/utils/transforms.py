import cv2
import imgaug.augmenters as iaa
import numpy as np


def order_points(pts):
    pts = np.array(pts)
    y_argmin = pts[:, 1].argmax()

    rect = np.zeros((4, 2), dtype='float32')

    if (
        np.linalg.norm(pts[y_argmin] - pts[(y_argmin + 1) % 4]) >
        np.linalg.norm(pts[y_argmin] - pts[(y_argmin - 1) % 4])
    ):
        for i in range(4):
            rect[i] = pts[(y_argmin + i + 2) % 4]
    else:
        for i in range(4):
            rect[i] = pts[(y_argmin + i + 1) % 4]

    if rect[0][0] > rect[1][0]:
        rect = rect[[3, 2, 1, 0]]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect

    len1, len2 = np.linalg.norm(tl - tr), np.linalg.norm(tl - bl)
    max_width = max(int(len1), int(len2))
    max_height = min(int(len1), int(len2))

    dst = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]], dtype='float32')

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


ocr_synthetic_transforms = iaa.Sequential([
    iaa.Resize({'height': 64, 'width': 320}),
    iaa.PerspectiveTransform(scale=(0.01, 0.05)),
    iaa.Sometimes(0.5, iaa.Multiply((0.5, 1.5))),
    iaa.Sometimes(0.5, iaa.JpegCompression(compression=(70, 99))),
    iaa.Affine(
        scale={'x': (0.95, 1.02), 'y': (0.95, 1.02)},
        translate_percent={'x': (-0.02, 0.02), 'y': (-0.02, 0.02)},
        rotate=(-3, 3),
        shear=(-5, 5),
    )
])

pad_transform = iaa.PadToAspectRatio(1.0, position='center')

ocr_transforms = iaa.Sequential([
    iaa.Resize({"height": 64, "width": 320}),
])
