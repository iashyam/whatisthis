import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, NLLLoss
from models import SimpleCNN
from models import BurrahMobileNet
from Train import Trainer
from ETL import Extractor, ImageDataset
from pathlib import Path
import mlflow

batch_size = 73
learning_rate = 0.01
epochs = 50

datas, labels = Extractor(Path("Data/cifar-100-python/test")).extract()
customdataset = ImageDataset(datas, labels)
ata, sample_label = customdataset[0]
dataloader = DataLoader(customdataset, batch_size=73, shuffle=True)
device = "cpu"
model = SimpleCNN()
adam =  Adam(lr=learning_rate, params=model.parameters())
sgd = SGD(lr=learning_rate, params=model.parameters())
loss_fn =  CrossEntropyLoss()
# trainer.train_loop(n_epochs=10, train_dataloader=dataloader, test_dataloader=dataloader)
mlflow.set_experiment("debug-pytorch-mlflow")
optims = {"sgd": sgd, "adam": adam}

for optim_name, optim_fn in optims.items():
    trainer = Trainer(model,loss_fn,optim_fn, device)
    with mlflow.start_run():
        mlflow.log_param("optimizer", optim_name)
        for epoch in range(epochs):
            loss, acc = trainer.training_step(epoch, dataloader)
            
            print("epoch", epoch)
            mlflow.log_metric("loss", loss.item(), step=epoch)
            mlflow.log_metric("accuracy", acc.item(), step=epoch)
            print(f"{epoch=}, {loss=},{acc=}")
