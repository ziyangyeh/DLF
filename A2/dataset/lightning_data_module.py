from typing import Optional, Union

import numpy as np
import pandas as pd
from lightning import LightningDataModule
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .bird_dataset import BirdDataset, BirdDatasetTriplet
from utils import image_aug

class LitDataModule(LightningDataModule):
    def __init__(
        self,
        data_cfg: OmegaConf,
    ):
        super(LitDataModule, self).__init__()
        self.save_hyperparameters()
        self.dataset = BirdDatasetTriplet if data_cfg.dataset.triplet else BirdDataset
        self.dataframe = pd.read_csv(data_cfg.dataset.csv_path)
        self.label_map = {k: idx for idx, k in enumerate(np.sort(np.loadtxt(data_cfg.dataset.label_map, dtype=str)))}
        self.image_size = data_cfg.dataset.image_size
        self.batch_size = data_cfg.dataloader.batch_size
        self.num_workers = data_cfg.dataloader.num_workers
        self.transform = image_aug(data_cfg.dataset.image_size, data_cfg.dataset.aug_level) if data_cfg.dataset.transform else None
        # self.save_hyperparameters(ignore=["data_cfg"])
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
                # transforms=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset(
                dataframe=self.dataframe,
                label_map=self.label_map,
                image_size=self.image_size,
                # transforms=self.transform,
            )
    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)
    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)
    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)
    def _dataloader(
        self,
        dataset: Union[BirdDataset, BirdDatasetTriplet],
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
    data_cfg = OmegaConf.load("config/default.yaml")
    data_cfg.data.dataset.triplet = True
    print(data_cfg)
    litdm = LitDataModule(data_cfg.data)
    litdm.setup()
    dl = litdm.train_dataloader()
    batch = next(iter(dl))
    print(batch.keys())
    print(batch["image"].shape)
    print(batch["positive"].shape)
    print(batch["negative"].shape)
    print(batch["label"].shape)
    print(batch["species"])

    litdm = LitDataModule(data_cfg.data)
    litdm.setup()
    dl = litdm.train_dataloader()
    batch = next(iter(dl))
    print(batch.keys())
    print(batch["image"].shape)
    print(batch["label"].shape)
    print(batch["species"])
