import numpy as np
import os
import pandas as pd
import sys
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RAFDataset(Dataset):
    def __init__(self, directory_path: str, subset: str, transform=None):
        if subset not in ("train", "test"):
            raise ValueError("Subset not valid!")

        if sys.version_info < (3, 6):
            raise SystemError("You must use Python version >= 3.6")

        self.directory_path = directory_path
        self.subset = subset
        self.transform = transform
        self.emotions = ["neutral", "happy", "surprise",
                         "sad", "anger", "disgust", "fear", "contempt"]

        key_map = {1: 2, 2: 6, 3: 5, 4: 1, 5: 3, 6: 4, 7: 0}

        self.csv_data = pd.read_csv(
            f"{directory_path}/EmoLabel/list_patition_label.txt", header=None, delim_whitespace=True)
        self.csv_data = self.csv_data[self.csv_data[0].str.startswith(subset)]
        self.csv_data[1] = self.csv_data[1].replace(
            list(key_map.keys()), list(key_map.values()))

        self.csv_data = self.csv_data.to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data[idx]
        filename, file_extension = os.path.splitext(
            f"{self.directory_path}/Image/aligned/{row[0]}")
        img = Image.open(f"{filename}_aligned{file_extension}")
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, row[1]
    
    def get_labels(self):
        return [row[1] for row in self.csv_data]
