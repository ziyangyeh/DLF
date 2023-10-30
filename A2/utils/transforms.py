from typing import Optional, Literal
import albumentations as A
from albumentations.pytorch import ToTensorV2

def image_aug(image_size: Optional[int] = None, level: Literal["easy", "medium", "hard"] = "easy"):
    if level == "easy":
        pre = [
            A.Flip(),
            A.OneOf(
                [
                    A.ShiftScaleRotate(),
                    A.SafeRotate(),
                ]
            ),
            A.OneOf([A.PixelDropout(), A.CoarseDropout()]),
            A.OneOf(
                [
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
                    A.HueSaturationValue(),
                    A.ColorJitter(),
                ],
            ),
            A.RandomBrightnessContrast(),
        ]
    elif level == "medium":
        pre = [
            A.Flip(),
            A.Transpose(),
            A.OneOf(
                [
                    A.ShiftScaleRotate(),
                    A.SafeRotate(),
                ]
            ),
            A.OneOf([A.PixelDropout(), A.CoarseDropout()]),
            A.OneOf(
                [
                    A.ISONoise(),
                    A.GaussNoise(),
                ],
                p=0.1,
            ),
            A.OneOf(
                [
                    A.MotionBlur(),
                    A.MedianBlur(blur_limit=3),
                    A.Blur(blur_limit=3),
                ],
                p=0.1,
            ),
            A.OneOf(
                [
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
                    A.HueSaturationValue(),
                    A.ColorJitter(),
                ],
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                ],
            ),
        ]
    else:
        pre = [
            A.Flip(),
            A.Transpose(),
            A.OneOf(
                [
                    A.ShiftScaleRotate(),
                    A.SafeRotate(),
                ]
            ),
            A.OneOf([A.PixelDropout(), A.CoarseDropout()]),
            A.OneOf(
                [
                    A.OpticalDistortion(),
                    A.GridDistortion(),
                    A.PiecewiseAffine(),
                ],
                p=0.1,
            ),
            A.OneOf(
                [
                    A.ISONoise(),
                    A.GaussNoise(),
                ],
                p=0.1,
            ),
            A.OneOf(
                [
                    A.MotionBlur(),
                    A.MedianBlur(blur_limit=3),
                    A.Blur(blur_limit=3),
                ],
                p=0.1,
            ),
            A.OneOf(
                [
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
                    A.HueSaturationValue(),
                    A.ColorJitter(),
                ],
            ),
            A.OneOf(
                [
                    A.RandomFog(),
                    A.RandomRain(),
                    A.RandomSnow(),
                ],
                p=0.1,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                ],
            ),
        ]

    post = [A.Normalize(), ToTensorV2()]
    if image_size:
        _trans = [
            A.CenterCrop(*(image_size, image_size), p=0.5),
            A.RandomResizedCrop(*(image_size, image_size), p=0.5),
            A.Resize(*(image_size, image_size)),
        ]
        post = _trans + post
    transforms = A.Compose(pre + post)
    return transforms
