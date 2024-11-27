import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from timm.data.mixup import Mixup

from model_factory import ModelFactory


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    args = parser.parse_args()
    return args


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    args: argparse.ArgumentParser,
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return validation_loss

def main():
    """Default Main Function."""
    # options
    args = opts()

    # Check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Updated device setup
    torch.manual_seed(args.seed)

    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    print(device)

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # load model and transform
    model, data_transform_train = ModelFactory(args.model_name).get_all(mode='train')
    data_transform_val = ModelFactory(args.model_name).get_all(mode='val')
    model.to(device)

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/train_images", transform=data_transform_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/val_images", transform=data_transform_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    mixup = Mixup(mixup_alpha=0.4, cutmix_alpha=0.4, label_smoothing=0.1)
    for images, labels in train_loader:
        images, labels = mixup(images, labels)

    # Setup optimizer for each model
    if args.model_name == "resnet50":
        # SGD
        optimizer = torch.optim.SGD([{'params': model.model.fc.parameters(), 'lr': 0.005},
        {'params': model.model.layer4.parameters(), 'lr': 0.002},
        {'params': model.model.layer3.parameters(), 'lr': 0.0008}], momentum=0.9, weight_decay=5e-4)
        # Cosine scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-5)

    elif args.model_name == "convnext":
        # Adam
        optimizer = optim.Adam([{'params': model.custom_classifier.parameters(), 'lr': 0.001}],weight_decay=1e-4)
        # Exponential scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    else:
        raise ValueError(f"Unknown model name: {args.model_name}")


    # Track validation loss evolution
    val_loss_evo = []

    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(1, args.epochs + 1):
        # Adjust learning rates for unfrozen layers

        if args.model_name == 'resnet50':
            # Unfreeze layer 4 when we reach epoch (15 here)
            if epoch == 15:
                for param in model.model.layer4.parameters():
                    param.requires_grad = True
                print("Unfroze Layer4")

            # Unfreeze layer 3 when we reach epoch (15 here)
            if epoch == 25:
                for param in model.model.layer3.parameters():
                    param.requires_grad = True
                print("Unfroze Layer3")

        # Print learning rate
        print(f"Epoch {epoch}: Learning Rates")
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"  Param Group {i}: {param_group['lr']}")

        # training loop and validation
        train(model, optimizer, train_loader, device, epoch, args)
        val_loss = validation(model, val_loader, device)

        # Append current loss
        val_loss_evo.append(val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = args.experiment + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)

        # also save the model every epoch
        model_file = args.experiment + "/model_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )
    # Save validation loss evolution to a .txt file
    with open(os.path.join(args.experiment, "validation_loss_evo.txt"), "w") as f:
        for epoch, loss in enumerate(val_loss_evo, 1):
            f.write(f"Epoch {epoch}: {loss:.4f}\n")
    print("Validation loss evolution saved to validation_loss_evo.txt")


if __name__ == "__main__":
    main()