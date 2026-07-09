import torch.nn as nn
from torchvision import models

def create_model(pretrained=False):
    base = models.mobilenet_v2(weights=None)

    in_features = base.classifier[1].in_features

    base.classifier = nn.Identity()

    model = nn.Sequential(
        base.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )

    return model