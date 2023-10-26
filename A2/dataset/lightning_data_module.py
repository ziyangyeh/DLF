from typing import Optional, Union

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .mushroom_dataset import MushroomDataset, MushroomDatasetTriplet


def image_aug(image_size: Optional[int] = None):
    pre = [
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        # A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    post = [A.Normalize(), ToTensorV2()]
    if image_size:
        post.insert(0, A.Resize(*(image_size, image_size)))
    transforms = A.Compose(pre + post)
    return transforms


class LitDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: OmegaConf,
    ):
        super(LitDataModule, self).__init__()
        self.dataset = (
            MushroomDatasetTriplet if cfg.dataset.triplet else MushroomDataset
        )
        self.dataframe = pd.read_csv(cfg.dataset.csv_path)
        self.label_map = {
            k: idx for idx, k in enumerate(np.loadtxt(cfg.dataset.label_map, dtype=str))
        }
        self.image_size = cfg.dataset.image_size
        self.batch_size = cfg.dataloader.batch_size
        self.num_workers = cfg.dataloader.num_workers
        self.transform = (
            image_aug(cfg.dataset.image_size) if cfg.dataset.transform else None
        )

        self.save_hyperparameters()
        # self.save_hyperparameters(ignore=["cfg"])

    def get_num_classes(self):
        return len(self.label_map)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(
                dataframe=self.dataframe,
                label_map=self.label_map,
                train_val="train",
                image_size=self.image_size,
                transforms=self.transform,
            )
            self.val_dataset = self.dataset(
                dataframe=self.dataframe,
                label_map=self.label_map,
                train_val="val",
                image_size=self.image_size,
                transforms=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset(
                dataframe=self.dataframe,
                label_map=self.label_map,
                image_size=self.image_size,
                transforms=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(
        self,
        dataset: Union[MushroomDataset, MushroomDatasetTriplet],
        train: Optional[bool] = None,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if train else False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if train else False,
        )


if __name__ == "__main__":
    cfg = OmegaConf.load("config/default.yaml")
    cfg.data.dataset.triplet = True
    print(cfg)
    litdm = LitDataModule(cfg.data)
    litdm.setup()
    dl = litdm.train_dataloader()
    batch = next(iter(dl))
    print(batch.keys())
    print(batch["image"].shape)
    print(batch["positive"].shape)
    print(batch["negative"].shape)
    print(batch["label"].shape)
    print(batch["species"])

    litdm = LitDataModule(cfg.data)
    litdm.setup()
    dl = litdm.train_dataloader()
    batch = next(iter(dl))
    print(batch.keys())
    print(batch["image"].shape)
    print(batch["label"].shape)
    print(batch["species"])
