
from abc import ABC, abstractmethod

from typing import Any, Tuple, Type

class LayerFactoryBase(ABC):
    """Abstract base class for layer factory classes."""

    @staticmethod
    @abstractmethod
    def conv2d(
        in_channels: int, out_channels: int, kernel_size: int,
        stride: int = 1, padding: int = 0, bias: bool = True
    ):
        pass

    @staticmethod
    @abstractmethod
    def conv1d(
        in_channels: int, out_channels: int, kernel_size: int,
        stride: int = 1, padding: int = 0, bias: bool = True
    ):
        pass

    @staticmethod
    @abstractmethod
    def batch_norm(num_features: int):
        pass

    @staticmethod
    @abstractmethod
    def relu():
        pass

    @staticmethod
    @abstractmethod
    def leaky_relu(negative_slope: float = 0.01):
        pass

    @staticmethod
    @abstractmethod
    def sigmoid():
        pass

    @staticmethod
    @abstractmethod
    def tanh():
        pass

    @staticmethod
    @abstractmethod
    def softmax(dim: int = -1):
        pass

    @staticmethod
    @abstractmethod
    def max_pool2d(kernel_size: int, stride: int = None):
        pass

    @staticmethod
    @abstractmethod
    def adaptive_avg_pool2d(output_size: Tuple[int, int]):
        pass

    @staticmethod
    @abstractmethod
    def linear(in_features: int, out_features: int, bias: bool = True):
        pass

    @staticmethod
    @abstractmethod
    def flatten():
        pass

    @staticmethod
    @abstractmethod
    def dropout(p: float = 0.5):
        pass

    @staticmethod
    @abstractmethod
    def lstm(
        input_size: int, hidden_size: int, num_layers: int = 1,
        batch_first: bool = True, dropout: float = 0.0
    ):
        pass

    @staticmethod
    @abstractmethod
    def gru(
        input_size: int, hidden_size: int, num_layers: int = 1,
        batch_first: bool = True, dropout: float = 0.0
    ):
        pass

    @staticmethod
    @abstractmethod
    def embedding(num_embeddings: int, embedding_dim: int):
        pass

    @staticmethod
    @abstractmethod
    def layer_norm(normalized_shape: int):
        pass

