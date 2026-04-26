from torch.utils.data import DataLoader
import torch
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from models import SimpleCNN
from Train import Trainer
from ETL import Extractor, ImageDataset
from pathlib import Path
import mlflow

batch_size = 73
learning_rate = 0.01
epochs = 50

datas, labels = Extractor(Path("Data/cifar-10-batches-py/data_batch_3")).extract()
customdataset = ImageDataset(datas, labels)
data, sample_label = customdataset[0]
dataloader = DataLoader(customdataset, batch_size=batch_size, shuffle=True)
device = "mps" if torch.mps.is_available() else "cpu"
print(device)
loss_fn =  CrossEntropyLoss()
# trainer.train_loop(n_epochs=10, train_dataloader=dataloader, test_dataloader=dataloader)
mlflow.set_experiment("Complex model for CIFAR-10 on mac")
mlflow.enable_system_metrics_logging()

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
            mlflow.log_metrics(
                {
                    "loss": loss,
                    "accuracy": acc,
                },
                step=epoch
            )

    mlflow.pytorch.log_model(model, name="mildly complex model")
