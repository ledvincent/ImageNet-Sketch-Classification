import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, convnext_large, ConvNeXt_Large_Weights

class ResNet50Classifier(nn.Module):
    def __init__(self, nclasses=1000):
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