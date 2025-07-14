
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow import keras # type: ignore

    

class TensorFlowLayers:
    """Factory class for creating TensorFlow layers"""
    
    @staticmethod
    def conv2d(in_channels: int, out_channels: int, kernel_size: int,
               stride: int = 1, padding: int = 0, bias: bool = True):
        return keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding='same' if padding > 0 else 'valid',
            use_bias=bias
        )

    @staticmethod
    def conv1d(in_channels: int, out_channels: int, kernel_size: int,
               stride: int = 1, padding: int = 0, bias: bool = True):
        return keras.layers.Conv1D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding='same' if padding > 0 else 'valid',
            use_bias=bias
        )

    @staticmethod
    def batch_norm(num_features: int):
        return keras.layers.BatchNormalization()

    @staticmethod
    def relu():
        return keras.layers.ReLU()

    @staticmethod
    def leaky_relu(negative_slope: float = 0.01):
        return keras.layers.LeakyReLU(alpha=negative_slope)

    @staticmethod
    def sigmoid():
        return keras.layers.Activation('sigmoid')

    @staticmethod
    def tanh():
        return keras.layers.Activation('tanh')

    @staticmethod
    def softmax(dim: int = -1):
        return keras.layers.Softmax(axis=dim)

    @staticmethod
    def max_pool2d(kernel_size: int, stride: Optional[int] = None):
        if stride is None:
            stride = kernel_size
        return keras.layers.MaxPooling2D(pool_size=kernel_size, strides=stride)

    @staticmethod
    def adaptive_avg_pool2d(output_size: Tuple[int, int]):
        if output_size == (1, 1):
            return keras.layers.GlobalAveragePooling2D()
        raise NotImplementedError("Adaptive pooling with custom output size not implemented for TensorFlow")

    @staticmethod
    def linear(in_features: int, out_features: int, bias: bool = True):
        return keras.layers.Dense(out_features, use_bias=bias)

    @staticmethod
    def flatten():
        return keras.layers.Flatten()

    @staticmethod
    def dropout(p: float = 0.5):
        return keras.layers.Dropout(rate=p)

    @staticmethod
    def lstm(input_size: int, hidden_size: int, num_layers: int = 1,
             batch_first: bool = True, dropout: float = 0.0):
        return keras.layers.LSTM(
            hidden_size,
            dropout=dropout,
            return_sequences=num_layers > 1
        )

    @staticmethod
    def gru(input_size: int, hidden_size: int, num_layers: int = 1,
            batch_first: bool = True, dropout: float = 0.0):
        return keras.layers.GRU(
            hidden_size,
            dropout=dropout,
            return_sequences=num_layers > 1
        )

    @staticmethod
    def embedding(num_embeddings: int, embedding_dim: int):
        return keras.layers.Embedding(num_embeddings, embedding_dim)

    @staticmethod
    def layer_norm(normalized_shape: int):
        return keras.layers.LayerNormalization()