import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FERPlusDataset(Dataset):
    def __init__(self, directory_path: str, subset: str, transform=None):
        inner_path_dict = {
            "train": "FER2013Train",
            "val": "FER2013Valid",
            "test": "FER2013Test",
        }

        self.inner_path = inner_path_dict.get(subset, None)
        if self.inner_path is None:
            raise ValueError("Subset not valid!")

        self.directory_path = directory_path
        self.subset = subset
        self.csv_data = pd.read_csv(
            f"{directory_path}/data/{self.inner_path}/label.csv", header=None).to_numpy()
        self.csv_data = self.csv_data[:, :-2]
        self.emotions = ["neutral", "happy", "surprise", "sad", "anger", "disgust", "fear", "contempt"]
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data[idx]
        img = Image.open(
            f"{self.directory_path}/{self.inner_path}/{row[0]}")
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        return img, np.argmax(row[2:])
