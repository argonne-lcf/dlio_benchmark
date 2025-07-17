from abc import ABC, abstractmethod

from typing import Any, Optional, Tuple, Type


class LayerFactoryBase(ABC):
    """Abstract base class for layer factory classes."""

    @abstractmethod
    def compute(self, input_data, target):
        pass


    @abstractmethod
    def set_optimizer(self, optimizer):
        pass

    @abstractmethod
    def conv2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        pass

    @abstractmethod
    def conv1d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        pass

    @abstractmethod
    def batch_norm(self, num_features: int):
        pass

    @abstractmethod
    def relu(
        self,
    ):
        pass

    @abstractmethod
    def leaky_relu(self, negative_slope: float = 0.01):
        pass

    @abstractmethod
    def sigmoid(
        self,
    ):
        pass

    @abstractmethod
    def tanh(
        self,
    ):
        pass

    @abstractmethod
    def softmax(self, dim: int = -1):
        pass

    @abstractmethod
    def max_pool2d(self, kernel_size: int, stride: Optional[int] = None):
        pass

    @abstractmethod
    def adaptive_avg_pool2d(self, output_size: Tuple[int, int]):
        pass

    @abstractmethod
    def linear(self, in_features: int, out_features: int, bias: bool = True):
        pass

    @abstractmethod
    def flatten(
        self,
    ):
        pass

    @abstractmethod
    def dropout(self, p: float = 0.5):
        pass

    @abstractmethod
    def lstm(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
    ):
        pass

    @abstractmethod
    def gru(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
    ):
        pass

    @abstractmethod
    def embedding(self, num_embeddings: int, embedding_dim: int):
        pass

    @abstractmethod
    def layer_norm(self, normalized_shape: int):
        pass
