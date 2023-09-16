from typing import Any, Dict

import pandas as pd
from torch.utils.data import Dataset


class DiabetesDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        super(DiabetesDataset, self).__init__()
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index) -> Dict[str, Any]:
        x = self.dataframe[self.dataframe.columns.values[:-1]].iloc[index].values
        labels = self.dataframe[self.dataframe.columns.values[-1]].iloc[index]
        return {"x": x, "labels": labels}


if __name__ == "__main__":
    raw = pd.read_csv("A1/data/raw/diabetes.csv")
    ds = DiabetesDataset(raw)
    print(ds[0])
