
from typing import Optional, Tuple
import torch
import torch.nn as nn



class PyTorchLayers:
    """Factory class for creating PyTorch layers"""
    
    @staticmethod
    def conv2d(in_channels: int, out_channels: int, kernel_size: int,
               stride: int = 1, padding: int = 0, bias: bool = True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=bias)

    @staticmethod
    def conv1d(in_channels: int, out_channels: int, kernel_size: int,
               stride: int = 1, padding: int = 0, bias: bool = True):
        return nn.Conv1d(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=bias)

    @staticmethod
    def batch_norm(num_features: int):
        return nn.BatchNorm2d(num_features)

    @staticmethod
    def relu():
        return nn.ReLU(inplace=True)

    @staticmethod
    def leaky_relu(negative_slope: float = 0.01):
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    @staticmethod
    def sigmoid():
        return nn.Sigmoid()

    @staticmethod
    def tanh():
        return nn.Tanh()

    @staticmethod
    def softmax(dim: int = -1):
        return nn.Softmax(dim=dim)

    @staticmethod
    def max_pool2d(kernel_size: int, stride: Optional[int] = None):
        if stride is None:
            stride = kernel_size
        return nn.MaxPool2d(kernel_size, stride=stride)

    @staticmethod
    def adaptive_avg_pool2d(output_size: Tuple[int, int]):
        return nn.AdaptiveAvgPool2d(output_size)

    @staticmethod
    def linear(in_features: int, out_features: int, bias: bool = True):
        return nn.Linear(in_features, out_features, bias=bias)

    @staticmethod
    def flatten():
        return nn.Flatten()

    @staticmethod
    def dropout(p: float = 0.5):
        return nn.Dropout(p=p)

    @staticmethod
    def lstm(input_size: int, hidden_size: int, num_layers: int = 1,
             batch_first: bool = True, dropout: float = 0.0):
        return nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                       batch_first=batch_first, dropout=dropout)

    @staticmethod
    def gru(input_size: int, hidden_size: int, num_layers: int = 1,
            batch_first: bool = True, dropout: float = 0.0):
        return nn.GRU(input_size, hidden_size, num_layers=num_layers,
                      batch_first=batch_first, dropout=dropout)

    @staticmethod
    def embedding(num_embeddings: int, embedding_dim: int):
        return nn.Embedding(num_embeddings, embedding_dim)

    @staticmethod
    def layer_norm(normalized_shape: int):
        return nn.LayerNorm(normalized_shape)