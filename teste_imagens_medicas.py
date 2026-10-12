import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from medmnist import PneumoniaMNIST
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             matthews_corrcoef, confusion_matrix)



DATA_ROOT = "./data"
IMG_SIZE = 128
NUM_CLASSES = 2

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """Fixa as sementes"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balance_dataset(dataset, seed = 42):
    rng = np.random.default_rng(seed)
    print("Random.default_rng: ", rng)
    labels = dataset.labels.flatten() # Lê rótulo, mas não carrega img

    idx_normal = np.where(labels == 0)[0]
    idx_pneumonia = np.where(labels == 1)[0]
    n = min(len(idx_normal), len(idx_pneumonia))

    indices = np.concatenate([
        rng.choice(idx_normal, n, replace=False),
        rng.choice(idx_pneumonia, n, replace=False)
    ])
    rng.shuffle(indices)
    return Subset(dataset, indices)

def get_labels(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset.labels.flatten()[dataset.indices]
    return dataset.labels.flatten()


def get_dataloaders(splits=("train", "val", "test"), batch_size=32, size=IMG_SIZE,
                    num_workers=0, balance=("train", "val", "test")):
    """Único ponto do curso que instancia o dataset.
    balance; splits que passam por undersampling (50/50). Os demais mantêm a distribuição
    original. Retorna {"train": loader, "val": loader e "teste": loader}"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0,5])
    ])

    os.makedirs(DATA_ROOT, exist_ok=True)
    loaders = {}

    for split in splits:
        dataset = PneumoniaMNIST(
            split=split,
            transform=transform,
            root=DATA_ROOT,
            size=size,
            download=True
        )
        if split in balance:
            dataset = balance_dataset(dataset)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),)
    print(type(loaders))
    print(loaders)
    return loaders

def Teacher(num_classes=NUM_CLASSES):
    """
    ResNet18

    """
    weights = ResNet18_Weights.DEFAULT
    teacher = models.resnet18(weights=weights)

    teacher.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    teacher.fc = nn.Linear(
        teacher.fc.in_features,
        num_classes
    )

    return teacher


class SmallCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_teacher(weights_path, device, num_classes=NUM_CLASSES):
    """Constrói a resNet18 e carrega os pesos treinados.
    O teacher fica congelado para a Knowledge Distillation"""

    teacher = Teacher(num_classes = num_classes)

    teacher.load_state_dict(
        torch.load(weights_path, map_location=device)
    )

    teacher = teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    return teacher

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def load_weights(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()

def load_student(weights_path, device):
    return load_weights(SmallCNN(), weights_path, device)


if __name__ == "__main__":
    print("")
    get_dataloaders()