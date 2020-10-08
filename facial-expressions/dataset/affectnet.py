import numpy as np
import pandas as pd
import sys
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AffectNetDataset(Dataset):
    def __init__(self, directory_path: str, subset: str, transform=None):
        inner_path_dict = {
            "train": "training",
            "val": "validation",
        }

        self.inner_path = inner_path_dict.get(subset, None)
        if self.inner_path is None:
            raise ValueError("Subset not valid!")

        if sys.version_info < (3, 6):
            raise SystemError("You must use Python version >= 3.6")

        self.directory_path = directory_path
        self.subset = subset
        self.transform = transform
        self.emotions = ["neutral", "happy", "surprise",
                         "sad", "anger", "disgust", "fear", "contempt"]

        self.csv_data = pd.read_csv(
            f"{directory_path}/Manually_Annotated_file_lists/{self.inner_path}.csv",
            header=None,
            usecols=[0, 1, 2, 3, 4, 6],
            skiprows=1 if subset == "train" else None)
        self.csv_data = self.csv_data[~self.csv_data[6].isin((8, 9, 10))]

        key_map = {2: 3, 3: 2, 4: 6, 6: 4}
        self.csv_data[6] = self.csv_data[6].replace(
            list(key_map.keys()), list(key_map.values()))
        self.csv_data = self.csv_data.to_numpy()

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data[idx]
        img = Image.open(
            f"{self.directory_path}/Manually_Annotated_Images/{row[0]}")
        img = img.crop((row[1], row[2], row[3], row[4]))
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, row[-1]
    
    def get_labels(self):
        return [row[-1] for row in self.csv_data]
