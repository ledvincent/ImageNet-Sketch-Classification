# ImageNet-Sketch Classification through Fine-Tuning

This repository contains an implementation of a fine-tuned ResNet50 model to classify ImageNet-Sketch images into 500 categories. The project demonstrates effective strategies for adapting pre-trained convolutional neural networks to a domain-specific dataset.

The original dataset can be found in this [GitHub repository](https://github.com/HaohanWang/ImageNet-Sketch).
This dataset is composed of 50000 images, 50 images for each of the 1000 ImageNet classes.

Before training, organize the dataset into three subfolders within a main directory named datasets. These subfolders should represent the training set (*train_set*), validation set (*val_set*), and test set (*test_set*), respectively. ('*data_set*' expected name to run the code)

---

## Project Files

### **1. model.py**
- Implements the `ResNet50Classifier`, which:
  - Adds fully connected layers for fine-tuning on the 1000 categories of ImageNet-Sketch.
  - Gradually unfreezes deeper layers during training for enhanced feature learning.
  - Uses dropout and additional dense layers for better generalization.

---

### **2. main.py**
- The main script for training the ResNet50 model.
- Key parameters:
  - `--data`: Path to the dataset. The dataset folder must contain 3 folders each corrresponding to the train, validation and test set, denoted by *train_set*, *val_set* and *test_set*
  - `--epochs`: Number of epochs for training (default: 40).
  - `--batch-size`: Batch size for training (default: 64).
  - `--lr`: Initial learning rate (default: 0.005).
  - `--model-name`: Model to train (`resnet50`).

- **Training Process**:
  - Fine-tuning begins by freezing all pre-trained ResNet50 layers except the added classification head.
  - Gradual unfreezing of deeper layers starts at epoch 15 (Layer 4) and epoch 30 (Layer 3).
  - Learning rates decrease progressively across the network using a cosine annealing scheduler.
  - Saves a copy of the model parameters (every *15* epochs by default)

---

### **3. data.py**
- Contains data preprocessing and augmentation functions tailored for ImageNet-Sketch.
- Transformations include:
  - Random resized cropping for uniform input size (224x224).
  - Random horizontal flips, rotation, and affine transformations for robustness.
  - Normalization using ImageNet mean and standard deviation.

---

### **4. model_factory.py**
- Facilitates modular model selection and data preprocessing pipelines.
- Supports easy addition of new models and transformations.

---

## Training Guide

### **Training the Model**
Run the following command to train the ResNet50 model:

```bash
python main.py --data <dataset_path> --epochs 40 --batch-size 64 --lr 0.005 --model-name resnet50
```

## Evaluation on test set

### Planned improvements:
Fine tuning a transformer model that was pre-trained on a vast corpus of diverse images.
