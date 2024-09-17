import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.models.FCNN import SimpleFCNN
from src.models.CNN import CNN
from src.utils.log import MetricLogger, ConfigLogger, ModelLogger, BestModelLogger


def train_one_epoch(cfg, manager, epoch, model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    for xx, yy in dataloader:
        xx = xx.to(device)
        yy = yy.to(device)
        optimizer.zero_grad()
        yy_hat = model(xx)
        loss = criterion(yy_hat, yy)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_loss = total_loss / num_batches
    manager.run(MetricLogger(cfg, epoch, epoch_loss))


def validate(cfg, manager, epoch, model, dataloader, criterion, device):
    model.eval()
    total_corrects = 0
    total_loss = 0.0
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    with torch.no_grad():
        for xx, yy in dataloader:
            yy_hat = model(xx)
            loss = criterion(yy_hat, yy)
            total_loss += loss.item()
            _, predicted = torch.max(yy_hat, 1)
            total_corrects += (predicted == yy).sum().item()
    epoch_loss = total_loss / num_batches
    accuracy = total_corrects / num_samples
    manager.run(MetricLogger(cfg, epoch, epoch_loss, accuracy))


def train_model(cfg, manager):
    # データの取得
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root=cfg.local.read_loc.rawdata_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root=cfg.local.read_loc.rawdata_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.hparam.train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.hparam.val_batch_size, shuffle=False)
    print("len(train_dataset): ", len(train_dataset))
    print("len(val_dataset): ", len(val_dataset))
    # ネットワークと学習設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.hparam.learning_rate)
    # 学習
    for epoch in range(cfg.hparam.num_epochs):
        print(epoch)
        train_one_epoch(cfg, manager, epoch, model, train_loader, criterion, optimizer, device)
        validate(cfg, manager, epoch, model, val_loader, criterion, device)
        manager.run(ModelLogger(cfg, epoch, model))
    # ログ
    manager.run(ConfigLogger(cfg))
    manager.run(BestModelLogger(cfg, model))
    return model