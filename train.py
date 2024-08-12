import os
import torch.optim as optim
from omegaconf import OmegaConf
from log import *
from utils.utils import *
from model.simpleFCNN import *
from data.data_loader import *


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


def evaluate(cfg, manager, epoch, model, dataloader, criterion):
    model.eval()
    total_corrects = 0
    total_loss = 0.0
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_corrects += (predicted == labels).sum().item()
    epoch_loss = total_loss / num_batches
    accuracy = total_corrects / num_samples
    manager.run(MetricLogger(cfg, epoch, epoch_loss, accuracy))


def train(cfg, manager):
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
        evaluate(cfg, manager, epoch, model, val_loader, criterion)
    # ログ
    manager.run(ConfigLogger(cfg))
    manager.run(BestModelLogger(cfg, model))
    return model


if __name__ == '__main__':
    # ワーキングディレクトリの設定
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    # 乱数設定、ログ設定  
    set_seed(42)
    # config読み込み
    cfg = OmegaConf.load("./config/config.yaml")
    # Log制御インスタンス
    manager = LoggerManager(enable_mlflow=False)
    # 学習
    train(cfg, manager)
