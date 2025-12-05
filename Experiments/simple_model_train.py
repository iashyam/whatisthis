import torch 

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 100, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # conv block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.Conv2d(32, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), #8->4
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # global pooling -> 1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 256*1*1 -> 256
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)    # logits
        )

        # optional simple weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


from ETL import Extractor, ImageDataset
from pathlib import Path
datas, labels = Extractor(Path("Data/cifar-100-python/test")).extract()
customdataset = ImageDataset(datas, labels)
sample_data, sample_label = customdataset[0]

from models import BurrahMobileNet
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, NLLLoss
dataloader = DataLoader(customdataset, batch_size=73, shuffle=True)
device = "cpu"
from Train import Trainer
model = SimpleCNN()
y_pred = model(sample_data.unsqueeze(0))
loss = CrossEntropyLoss()(y_pred, sample_label.unsqueeze(0))
trainer = Trainer(model, CrossEntropyLoss(), Adam(params=model.parameters(), lr=0.01), device)
trainer.train_loop(n_epochs=10, train_dataloader=dataloader, test_dataloader=dataloader)
