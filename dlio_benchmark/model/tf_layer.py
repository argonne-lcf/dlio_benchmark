from typing import Any, Optional, Tuple
import tensorflow as tf
from tensorflow import keras

from dlio_benchmark.model.layer import LayerFactoryBase  # type: ignore


class TensorFlowLayers(LayerFactoryBase):
    """Factory class for creating TensorFlow layers"""

    def __init__(self, loss_function, communication: bool = False):
        super().__init__()
        self._model = None 
        self._optimizer = None
        self._loss_function = loss_function
        self._layer_registry = {}  # Track created layers similar to PyTorch
        self.communication = communication

    def _register_layer(self, layer: keras.layers.Layer, name: Optional[str] = None) -> keras.layers.Layer:
        """Register a layer for automatic model building"""
        if name is None:
            name = f"layer_{len(self._layer_registry)}"
        self._layer_registry[name] = layer
        return layer

    def conv2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        layer = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="same" if padding > 0 else "valid",
            use_bias=bias,
        )
        return self._register_layer(layer)

    def conv1d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        layer = keras.layers.Conv1D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="same" if padding > 0 else "valid",
            use_bias=bias,
        )
        return self._register_layer(layer)

    def batch_norm(self, num_features: int):
        layer = keras.layers.BatchNormalization()
        return self._register_layer(layer)

    def relu(self):
        layer = keras.layers.ReLU()
        return self._register_layer(layer)

    def leaky_relu(self, negative_slope: float = 0.01):
        layer = keras.layers.LeakyReLU(alpha=negative_slope)
        return self._register_layer(layer)

    def sigmoid(self):
        layer = keras.layers.Activation("sigmoid")
        return self._register_layer(layer)

    def tanh(self):
        layer = keras.layers.Activation("tanh")
        return self._register_layer(layer)

    def softmax(self, dim: int = -1):
        layer = keras.layers.Softmax(axis=dim)
        return self._register_layer(layer)

    def max_pool2d(self, kernel_size: int, stride: Optional[int] = None):
        if stride is None:
            stride = kernel_size
        layer = keras.layers.MaxPooling2D(pool_size=kernel_size, strides=stride)
        return self._register_layer(layer)

    def adaptive_avg_pool2d(self, output_size: Tuple[int, int]):
        if output_size == (1, 1):
            layer = keras.layers.GlobalAveragePooling2D()
        else:
            raise NotImplementedError(
                "Adaptive pooling with custom output size not implemented for TensorFlow"
            )
        return self._register_layer(layer)

    def linear(self, in_features: int, out_features: int, bias: bool = True):
        layer = keras.layers.Dense(out_features, use_bias=bias)
        return self._register_layer(layer)

    def flatten(self):
        layer = keras.layers.Flatten()
        return self._register_layer(layer)

    def dropout(self, p: float = 0.5):
        layer = keras.layers.Dropout(rate=p)
        return self._register_layer(layer)

    def lstm(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
    ):
        layer = keras.layers.LSTM(
            hidden_size, dropout=dropout, return_sequences=num_layers > 1
        )
        return self._register_layer(layer)

    def gru(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
    ):
        layer = keras.layers.GRU(
            hidden_size, dropout=dropout, return_sequences=num_layers > 1
        )
        return self._register_layer(layer)

    def embedding(self, num_embeddings: int, embedding_dim: int):
        layer = keras.layers.Embedding(num_embeddings, embedding_dim)
        return self._register_layer(layer)

    def layer_norm(self, normalized_shape: int):
        layer = keras.layers.LayerNormalization()
        return self._register_layer(layer)
    def get_model(self, forward_fn: Any) -> keras.Model:
        """
        Constructs and returns a Keras Model.
        Args:
            forward_fn: The function that defines the forward pass of the model.
                        This function will be used as the `call` method of the Keras Model.
        Returns:
            A Keras Model instance.
        """
        if self._model is not None:
            return self._model

        class ModelWrapper(keras.Model):
            """
            A Keras Model subclass that wraps the forward pass and tracks layers.
            It uses the provided `forward_fn` for its `call` method.
            """
            def __init__(self, layer_registry):
                super().__init__()
                self._layer_dict = {}

                # Register all layers from the factory's registry as attributes of this model
                for name, layer in layer_registry.items():
                    setattr(self, name, layer)
                    self._layer_dict[name] = layer

            def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
                """
                The forward pass method for the Keras Model.
                It calls the provided `forward_fn`.
                """
                # The forward_fn is expected to be a partial function (e.g., partial(self.forward, self))
                # where the first 'self' is the actual model instance (like ResNet50).
                # So, simply calling forward_fn with inputs is correct.
                return forward_fn(inputs)

            def get_layer_dict(self):
                """Returns the dictionary of registered layers."""
                return self._layer_dict

            def list_layers(self):
                """Prints all registered layers."""
                print("Registered layers:")
                for name, layer in self._layer_dict.items():
                    print(f"  {name}: {type(layer).__name__}")

        # Instantiate the ModelWrapper
        self._model = ModelWrapper(self._layer_registry)
        if self.communication:
            import horovod.tensorflow as hvd
            hvd.init()
            self._model = hvd.DistributedOptimizer(self._model)

        # layer_dict = self._model.get_layer_dict()
        # print(f"Registered layers: {list(layer_dict.keys())}")

        # print("Trainable variables:")
        # if not self._model.trainable_variables:
        #     print("  No trainable variables found yet. They will appear after the model processes its first input.")
        # for var in self._model.trainable_variables:
        #     print(f"  {var.name}: {var.shape}")

        return self._model


    def set_optimizer(self, optimizer, *args, **kwargs):
        assert self._model is not None, "Model must be set before optimizer."
        self._optimizer = optimizer(*args, **kwargs)
        self._model.compile(optimizer=self._optimizer)
    

    def compute(self, batch) -> None:
        assert self._model is not None, "Model must be set before compute step."
        assert self._optimizer is not None, "Optimizer must be set before compute step."

        # Perform the forward pass and backward pass within this function
        # This ensures the GradientTape correctly records all operations.
        # print Input shape of _model 

        #TODO: Remove hardcoding and integrate with ray PR
        if isinstance(batch, tf.Tensor):
            if len(batch.shape) == 3:
                # Duplicate array thrice to make three channels
                batch = tf.expand_dims(batch, axis=1)
                batch = tf.repeat(batch, repeats=3, axis=1)
                # this gives shape (1,3, x,x)
                # I need shape (1, x, x, 3)
                batch = tf.transpose(batch, perm=[0, 2, 3, 1])
            elif len(batch.shape) == 2:
                batch = tf.reshape(batch, [1, *batch.shape, 1])
            
            # cast to float32
            input_data = tf.cast(batch, tf.float32)
            batch = tf.cast(batch, tf.float32)
            target = tf.zeros((input_data.shape[0],1000))
        else: 
            input_data, target = batch

        if self.communication:
            import horovod.tensorflow as hvd
            tape = hvd.DistributedGradientTape()
        else:
            tape = tf.GradientTape()
        with tape:
            # 1. Forward pass
            pred = self._model(input_data, training=True)

            # Convert target to tensor if needed
            target = tf.convert_to_tensor(target)

            # 2. Calculate loss
            # TODO: Make loss configurable
            loss = self._loss_function(target, pred)
            # loss = keras.losses.MeanSquaredError()(target, pred)

        # 3. Calculate gradients
        gradients = tape.gradient(loss, self._model.trainable_variables)

        # 4. Apply gradients (backward pass)
        filtered_gradients_and_vars = []
        for grad, var in zip(gradients, self._model.trainable_variables):
            if grad is not None:
                filtered_gradients_and_vars.append((grad, var))
            else:
                print(f"Warning: Gradient is None for variable: {var.name}. Skipping update for this variable.")

        if not filtered_gradients_and_vars:
            print("Warning: No valid gradients found to apply. Optimizer step skipped.")
        else:
            self._optimizer.apply_gradients(filtered_gradients_and_vars)
        

    
    def reset_model(self):
        """Reset the model to allow creating a new one"""
        self._model = None
        self._optimizer = None
        self._layer_registry = {}

    def get_layer_registry(self):
        """Return the current layer registry"""
        return self._layer_registry.copy()

    def list_registered_layers(self):
        """Print all registered layers in the factory"""
        print("Factory layer registry:")
        for name, layer in self._layer_registry.items():
            print(f"  {name}: {type(layer).__name__}")