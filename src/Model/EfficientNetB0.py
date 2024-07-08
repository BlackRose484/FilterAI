import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB0(nn.Module):
    def __init__(self, num_points):
        super(EfficientNetB0, self).__init__()
        self.num_points = num_points
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_points * 2)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.num_points, 2)
        return x
