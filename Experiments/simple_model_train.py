from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from models import SimpleCNN
from Train import Trainer
from ETL import Extractor, ImageDataset
from pathlib import Path
import mlflow
import os

batch_size = 73
learning_rate = 0.01
epochs = 25

print(f"cwd: {os.getcwd()}")

datas, labels = Extractor(Path("Data/cifar-10-batches-py/data_batch_4")).extract()
customdataset = ImageDataset(datas, labels)
data, sample_label = customdataset[0]
dataloader = DataLoader(customdataset, batch_size=73, shuffle=True)
device = "cpu"
loss_fn =  CrossEntropyLoss()
# trainer.train_loop(n_epochs=10, train_dataloader=dataloader, test_dataloader=dataloader)
mlflow.set_experiment("run for CIFR10- adam")
optims = {"sgd": lambda params: SGD(params),
          "adam": lambda params: Adam(params)}

optims = {"adam": lambda params: Adam(params)}

for optim_name, optim_fn in optims.items():
    model = SimpleCNN(num_classes=10)
    optim = optim_fn(model.parameters())
    trainer = Trainer(model,loss_fn,optim, device)
    with mlflow.start_run(run_name="complex+mocdel"):
        mlflow.log_param("optimizer", optim_name)
        for epoch in range(epochs):
            loss, acc = trainer.training_step(epoch, dataloader)
            
            print("epoch", epoch)
            mlflow.log_metric("loss", loss.item(), step=epoch)
            mlflow.log_metric("accuracy", acc.item(), step=epoch)
            print(f"{epoch=}, {loss=},{acc=}")
