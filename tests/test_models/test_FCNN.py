import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.FCNN import SimpleFCNN


class TestSimpleFCNN(unittest.TestCase):    
    def setUp(self):
        self.model = SimpleFCNN()

    def test_forward_shape(self):
        # 入力が与えられたときに、出力の形状が正しいかをテスト
        input_tensor = torch.randn(16, 1, 28, 28)  # バッチサイズ16のダミーデータ
        output_tensor = self.model(input_tensor)
        print("output tensor size is: ", output_tensor.shape)
        self.assertEqual(output_tensor.shape, (16, 10))  # 出力の形状を確認

    def test_parameter_count(self):
        # 学習パラメータ数が、理論値通りかをテスト
        expected_param_count = (
            (28 * 28 * 128) + 128 +  # fc1: 入力サイズ * 出力サイズ + バイアス
            (128 * 10) + 10  # fc2: 入力サイズ * 出力サイズ + バイアス
        )
        # 実際に計算した学習パラメータの合計数
        actual_param_count = sum(p.numel() for p in self.model.parameters())
        print("number of parameter is: ", actual_param_count)
        self.assertEqual(actual_param_count, expected_param_count)


if __name__ == '__main__':
    unittest.main()