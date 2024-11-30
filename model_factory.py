"""Python file to instantite the model and the transform that goes with it."""

from data import resnet_transforms
from model import ResNet50Classifier

class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model, self.transforms = self.init_model()

    # Get model and data transformations
    def get_all(self, mode: str):
        if self.model_name == "resnet50":
            if mode in ["train", "test"]:
                return self.model, self.transforms(mode)
            elif mode == 'val':
                return self.transforms(mode)
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
        raise ValueError(f"Invalid mode: {mode}. Use 'train', 'val', or 'test'.")

    # Return the chosen model
    def init_model(self):
        if self.model_name == "resnet50":
            return ResNet50Classifier(), resnet_transforms
        else:
            raise NotImplementedError("Model not implemented. Only 'resnet50'")