import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, NLLLoss
from models import SimpleCNN
from models import BurrahMobileNet
from Train import Trainer
from ETL import Extractor, ImageDataset
from pathlib import Path

datas, labels = Extractor(Path("Data/cifar-100-python/test")).extract()
customdataset = ImageDataset(datas, labels)
ata, sample_label = customdataset[0]
dataloader = DataLoader(customdataset, batch_size=73, shuffle=True)
device = "cpu"
model = SimpleCNN()
optimizer =  Adam(params=model.parameters())
loss_fn =  CrossEntropyLoss()
trainer = Trainer(model,loss_fn,optimizer, device)
trainer.train_loop(n_epochs=10, train_dataloader=dataloader, test_dataloader=dataloader)
