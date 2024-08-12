import unittest
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


class TestSimpleFCNN(unittest.TestCase):    
    def setUp(self):
        self.model = SimpleFCNN()

    def test_forward_shape(self):
        # 入力が与えられたときに、出力の形状が正しいかをテスト
        input_tensor = torch.randn(16, 1, 28, 28)  # バッチサイズ16のダミーデータ
        output_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape, (16, 10))  # 出力の形状を確認

    def test_parameter_count(self):
        # 学習パラメータ数が、理論値通りかをテスト
        # 手動で計算した学習パラメータの合計数
        expected_param_count = (
            (28 * 28 * 128) + 128 +  # fc1: 入力サイズ * 出力サイズ + バイアス
            (128 * 10) + 10  # fc2: 入力サイズ * 出力サイズ + バイアス
        )
        # 実際に計算した学習パラメータの合計数
        actual_param_count = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(actual_param_count, expected_param_count)


if __name__ == '__main__':
    unittest.main()