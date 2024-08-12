import torch
import torch.nn as nn


# 全結合のみからなる簡単なネットワーク
class SimpleFCNN(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, output_size=10):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)   # 例 [bsize, 1, 28, 28]->[bsize, 784]
        x = self.fc1(x)         # 例 [bsize, 784]->[bsize, 128]
        x = torch.relu(x)       # 例 [bsize, 128]->[bsize, 128]
        x = self.fc2(x)         # 例 [bsize, 128]->[bsize, 10]
        return x