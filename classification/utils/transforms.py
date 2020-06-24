import imgaug.augmenters as iaa

corrupt_transform = iaa.SomeOf((2, 3),
    [
        iaa.OneOf([
            iaa.Crop(percent=((0, 0.05), (0.2, 0.4), (0, 0.05), (0, 0.4))),
            iaa.Crop(percent=((0, 0.05), (0, 0.4), (0, 0.05), (0.2, 0.4))),
        ]),
        iaa.Sometimes(
            0.05,
            iaa.Rot90([1, 3])
        ),
        iaa.Affine(scale=(1.5, 2.0))
    ]
)
