import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, convnext_large, ConvNeXt_Large_Weights

class ResNet50Classifier(nn.Module):
    def __init__(self, nclasses=500):
        super(ResNet50Classifier, self).__init__()
        self.model = resnet50(weights='IMAGENET1K_V2')

        # Freeze layers 1 and 2
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1000),  # Additional hidden layer
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1000, nclasses)
        )

    def forward(self, x):
        return self.model(x)

class ConvNext(nn.Module):
    def __init__(self, nclasses=500):
        super(ConvNext, self).__init__()
        self.model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)

        # Freeze all layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Additional layers to tune during training
        self.custom_classifier = nn.Sequential(
            nn.Conv2d(1536, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, nclasses)
        )

    def forward(self, x):
        x = self.model.features(x)
        x = self.custom_classifier(x)
        return x