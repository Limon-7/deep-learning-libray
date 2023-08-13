import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "data"
print(DATA_DIR)

# Hyperparameters
IN_CHANNEL = 1
NUM_CLASSES = 10
NUM_EPOCHS = 3

# HYPERPARAMETER SEARCH
BATCH_SIZES = [32, 256]
LEARNING_RATES = [1e-2, 1e-3, 1e-4, 1e-5]
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

train_dataset = datasets.MNIST(
    root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=True
)


# def train_loader(batch_size=64):
#     train_loader = DataLoader(
#         dataset=train_dataset, batch_size=batch_size, shuffle=True
#     )
#     return train_loader


loss_fn = nn.CrossEntropyLoss()


def optimizer(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
