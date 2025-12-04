from torch.utils.data import Dataset
import torch
from utils import preprocess_image
from PIL import Image

class ImageDataset(Dataset):

    def __init__(self, datas, labels):
        super().__init__()
        self.datas = datas
        self.labels = torch.Tensor(labels)

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.datas[index])
        image = preprocess_image(image)
        return image, self.labels[index]