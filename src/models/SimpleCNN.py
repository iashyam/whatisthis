import torch 
from torch import nn

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
