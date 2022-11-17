from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch

class CrossDataset(Dataset):
    def __init__(self, label_file:str, img_path:str):
        super().__init__()
        self.labels = pd.read_csv(label_file)
        self.img_path = Path(img_path)


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        sample = self.labels.iloc[[index]][0].values
        img_name = sample[0]

        X = np.asarray(Image.open(self.path / img_name))
        y = sample[[1, 2]]
        return torch.tensor(X), torch.tensor(y)

