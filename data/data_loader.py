from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .dataset import *


def create_train_val_dataloader(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    full_dataset = datasets.MNIST(root="./data/raw", train=True, download=True, transform=transform)
    small_dataset = create_subset(full_dataset, cfg.dataset.subset_size_ratio)
    train_dataset, val_dataset = split_dataset(small_dataset, cfg.dataset.train_size_ratio)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.train_param.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.train_param.batch_size, shuffle=False)
    return train_loader, val_loader


def create_test_dataloader(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    full_dataset = datasets.MNIST(root="./data/raw", train=False, download=True, transform=transform)
    test_dataset = create_subset(full_dataset, cfg.dataset.subset_size_ratio)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.test_param.batch_size, shuffle=False)
    return test_loader
