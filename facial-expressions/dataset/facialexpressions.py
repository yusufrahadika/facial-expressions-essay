import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FacialExpressionsDataset(Dataset):
    def __init__(self, directory_path: str, transform=None):
        self.directory_path = directory_path
        self.transform = transform
        self.emotions = ["neutral", "happy", "surprise",
                         "sad", "anger", "disgust", "fear", "contempt"]
        self.csv_data = pd.concat([
            pd.read_csv(f"{directory_path}/data/legend.csv",
                        header=None, skiprows=[0]),
            pd.read_csv(
                f"{directory_path}/data/500_picts_satz.csv", header=None),
        ], ignore_index=True).drop([0], axis=1)
        self.csv_data[2] = self.csv_data[2].str.lower()
        self.csv_data[2] = self.csv_data[2].replace(
            ["happiness", "sadness"], ["happy", "sad"])
        self.csv_data = self.csv_data.to_numpy()

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data[idx]
        img = Image.open(f"{self.directory_path}/images/{row[0]}")
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, self.emotions.index(row[1])
    
    def get_labels(self):
        return [self.emotions.index(row[1]) for row in self.csv_data]
