from typing import Any, Callable, Dict, Literal, Optional

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_map: Dict[str, int],
        train_val: Optional[Literal["train", "val"]] = None,
        image_size: Optional[int] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super(BirdDataset, self).__init__()
        if train_val:
            flag = False if train_val == "train" else True
            self.dataframe = dataframe[dataframe["is_valid"] == flag]
            self.dataframe.reset_index(inplace=True, drop="index")
        else:
            self.dataframe = dataframe  # test dataset
        self.label_map = label_map
        self.image_size = image_size
        if transforms:
            self.transforms = transforms
        else:
            compose_list = [A.Normalize(), ToTensorV2()]
            if image_size:
                compose_list.insert(0, A.Resize(*(image_size, image_size)))
            self.transforms = A.Compose(compose_list)
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, index) -> Any:
        img_path, species = (
            self.dataframe["image"][index],
            self.dataframe["label"][index],
        )
        img = np.asarray(Image.open(img_path).convert("RGB"))
        H, W, C = img.shape
        if min(H, W) < self.image_size:
            img = A.SmallestMaxSize(self.image_size)(image=img)["image"]
        label = self.label_map[species]
        img = self.transforms(image=img)["image"]
        return {"image": img, "label": label, "species": species}

class BirdDatasetTriplet(BirdDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_map: Dict[str, int],
        train_val: Literal["train", "val"] | None = None,
        image_size: int | None = None,
        transforms: Callable[..., Any] | None = None,
    ) -> None:
        super(BirdDatasetTriplet, self).__init__(dataframe, label_map, train_val, image_size, transforms)
    def __getitem__(self, index) -> Any:
        result = super().__getitem__(index)
        species = result["species"]
        pos_inds = np.where(self.dataframe[self.dataframe["label"] == species].index.values != index)[0]
        neg_inds = self.dataframe[self.dataframe["label"] != species].index.values
        pos_selected = np.random.choice(pos_inds, size=1, replace=True)
        neg_selected = np.random.choice(neg_inds, size=1, replace=True)
        pos_path = self.dataframe["image"][pos_selected[0]]
        neg_path = self.dataframe["image"][neg_selected[0]]
        pos_img = np.asarray(Image.open(pos_path).convert("RGB"))
        neg_img = np.asarray(Image.open(neg_path).convert("RGB"))
        pos_img = self.transforms(image=pos_img)["image"]
        neg_img = self.transforms(image=neg_img)["image"]
        result["positive"] = pos_img
        result["negative"] = neg_img
        return result

if __name__ == "__main__":
    label_map = np.loadtxt("data/bird.txt", dtype=str)
    label_map = {k: idx for idx, k in enumerate(label_map)}
    df = pd.read_csv("data/bird.csv")
    ds = BirdDataset(df, label_map, "train")
    # print(ds[0]["image"].shape)
    ds_t = BirdDatasetTriplet(df, label_map, "train")
    print(ds_t[0]["image"].shape)
    print(ds_t[0]["positive"].shape)
    print(ds_t[0]["negative"].shape)
