import torch
from src.utils.log import MetricLogger


def evaluate_model(cfg, manager, epoch, model, dataloader, criterion):
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
