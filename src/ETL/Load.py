from torch.utils.data import Dataset
import pandas as pd
import torch
from utils import preprocess_image
from PIL import Image

class ImageDataset(Dataset):

    def __init__(self, datas, labels):
        super().__init__()
        self.labels = pd.get_dummies(pd.Series(labels)).to_numpy()
        self.datas = torch.Tensor(datas)
        self.labels = torch.Tensor(self.labels)

        
        print(f"{self.datas.shape=}, {self.labels.shape=}")

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]