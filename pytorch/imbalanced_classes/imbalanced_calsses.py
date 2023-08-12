import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from pathlib import Path


def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms) # A generic data loader where the images are arranged in this way by default
    subdirectories = dataset.classes
    class_weights = []

    # loop through each subdirectory and calculate the class weight
    # that is 1 / len(files) in that subdirectory
    for subdir in subdirectories:
        files = os.listdir(os.path.join(root_dir, subdir)) # create a list of files for each directory
        class_weights.append(1 / len(files)) # set weight for each class, ex- class_weights=[.02, 1]

    sample_weights = [0] * len(dataset) # create a empty list with of dataset length and fill each index with 0

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight # set weight for each of the datapoint



    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader


def main():
    path = Path(__file__).parent / "dataset"

    loader = get_loader(root_dir=path, batch_size=8)

    num_retrievers = 0
    num_elkhounds = 0
    for epoch in range(10):
        for data, labels in loader:
            num_retrievers += torch.sum(labels == 0)
            num_elkhounds += torch.sum(labels == 1)

    print(num_retrievers.item())
    print(num_elkhounds.item())


if __name__ == "__main__":
    main()
