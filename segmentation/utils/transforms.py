import imgaug.augmenters as iaa


def configure_transforms(resize_to):
    train_transforms = iaa.Sequential([
        iaa.PadToAspectRatio(1.0, position='center'),
        iaa.Resize({"height": resize_to, "width": resize_to}),
        iaa.Affine(
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-10, 10),
            shear=(-5, 5),
        ),
    ])

    valid_transforms = iaa.Sequential([
        iaa.PadToAspectRatio(1.0, position='center'),
        iaa.Resize({"height": resize_to, "width": resize_to}),
    ])
    return train_transforms, valid_transforms
