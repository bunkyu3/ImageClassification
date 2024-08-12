# import os
# from omegaconf import OmegaConf
# from log import *
# from utils.utils import *
# from model.simpleFCNN import *
# from data.data_loader import *
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.models.simpleFCNN import SimpleFCNN
from src.evaluation.evaluate_model import evaluate_model
from src.utils.log import MetricLogger, ConfigLogger, BestModelLogger
from src.utils.utils import create_subset, split_dataset


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


def train_one_epoch(cfg, manager, epoch, model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_loss = total_loss / num_batches
    manager.run(MetricLogger(cfg, epoch, epoch_loss))


def train_model(cfg, manager):
    # データの取得
    train_loader, val_loader = create_train_val_dataloader(cfg)
    # ネットワークと学習設定
    model = SimpleFCNN(input_size=28*28, hidden_size=128, output_size=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train_param.learning_rate)
    # 学習
    for epoch in range(cfg.train_param.num_epochs):
        print(epoch)
        train_one_epoch(cfg, manager, epoch, model, train_loader, criterion, optimizer)
        evaluate_model(cfg, manager, epoch, model, val_loader, criterion)
    # ログ
    manager.run(ConfigLogger(cfg))
    manager.run(BestModelLogger(cfg, model))
    return model