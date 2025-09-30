from typing import Any, Optional, Tuple
import torch
import torch.nn as nn



class PyTorchLayers:
    """Factory class for creating PyTorch layers"""

    def __init__(self, loss_function, communication: bool = False, gpu_id: int = -1) -> None:
        super().__init__()
        self._optimizer = None
        self._model = None
        self._loss_function = loss_function
        self._layer_registry = {}  # Track created layers
        self.communication = communication
        self.gpu_id = gpu_id

    def _register_layer(self, layer: nn.Module, name: Optional[str] = None) -> nn.Module:
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
        layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
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
        layer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        return self._register_layer(layer)

    def batch_norm(self, num_features: int):
        layer = nn.BatchNorm2d(num_features)
        return self._register_layer(layer)

    def relu(self):
        layer = nn.ReLU(inplace=True)
        return self._register_layer(layer)

    def leaky_relu(self, negative_slope: float = 0.01):
        layer = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        return self._register_layer(layer)

    def sigmoid(self):
        layer = nn.Sigmoid()
        return self._register_layer(layer)

    def tanh(self):
        layer = nn.Tanh()
        return self._register_layer(layer)

    def softmax(self, dim: int = -1):
        layer = nn.Softmax(dim=dim)
        return self._register_layer(layer)

    def max_pool2d(self, kernel_size: int, stride: Optional[int] = None):
        if stride is None:
            stride = kernel_size
        layer = nn.MaxPool2d(kernel_size, stride=stride)
        return self._register_layer(layer)

    def adaptive_avg_pool2d(self, output_size: Tuple[int, int]):
        layer = nn.AdaptiveAvgPool2d(output_size)
        return self._register_layer(layer)

    def linear(self, in_features: int, out_features: int, bias: bool = True):
        layer = nn.Linear(in_features, out_features, bias=bias)
        return self._register_layer(layer)

    def flatten(self):
        layer = nn.Flatten()
        return self._register_layer(layer)

    def dropout(self, p: float = 0.5):
        layer = nn.Dropout(p=p)
        return self._register_layer(layer)

    def lstm(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
    ):
        layer = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
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
        layer = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
        )
        return self._register_layer(layer)

    def embedding(self, num_embeddings: int, embedding_dim: int):
        layer = nn.Embedding(num_embeddings, embedding_dim)
        return self._register_layer(layer)

    def layer_norm(self, normalized_shape: int):
        layer = nn.LayerNorm(normalized_shape)
        return self._register_layer(layer)

    def get_model(self, forward_fn: Any, ) -> nn.Module:
        if self._model is not None:
            return self._model

        # If communication init process group
        if self.communication:
            from dlio_benchmark.utils.utility import DLIOMPI
            import torch.distributed as dist
            import socket
            import os
            rank = DLIOMPI.get_instance().rank()
            if rank == 0:
                master_addr = socket.gethostname()
            else:
                master_addr = None
            master_addr = DLIOMPI.get_instance().comm().bcast(master_addr, root=0)
            world_size = DLIOMPI.get_instance().size()
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(2345)
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)

        class Model(nn.Module):
            def __init__(self, layer_registry):
                super().__init__()
                # Register all layers from the factory
                for name, layer in layer_registry.items():
                    setattr(self, name, layer)
            
            def _register_layer_list(self, layer_list, list_name):
                """Register a list of layers or blocks"""
                for i, item in enumerate(layer_list):
                    if hasattr(item, 'conv1'):  # This is a ResNet block
                        self._register_block_layers(item, f"{list_name}_{i}")
                    elif isinstance(item, nn.Module):
                        setattr(self, f"{list_name}_{i}", item)
                    else:
                        print(f"Warning: {list_name} contains unsupported type: {type(item)}")
            
            def _register_block_layers(self, block, prefix):
                """Register all layers from a framework-agnostic block"""
                # Register individual layers from the block
                for attr_name in ['conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3', 'relu']:
                    if hasattr(block, attr_name):
                        layer = getattr(block, attr_name)
                        if isinstance(layer, nn.Module):
                            setattr(self, f"{prefix}_{attr_name}", layer)
                        else: 
                            print(f"Warning: {attr_name} in {prefix} is not a nn.Module: {type(layer)}")
                    else: 
                        print(f"Warning: {attr_name} not found in block {prefix}")
                
                # Handle downsample if it exists
                if hasattr(block, 'downsample') and block.downsample is not None:
                    if isinstance(block.downsample, nn.Module):
                        setattr(self, f"{prefix}_downsample", block.downsample)
                    elif callable(block.downsample):
                        # If downsample is a lambda/function, we need to register any layers it might use
                        # This is more complex, but we can try to extract from the function's closure
                        try:
                            # Get the closure variables (this works for lambdas that capture variables)
                            if hasattr(block.downsample, '__closure__') and block.downsample.__closure__:
                                closure_vars = block.downsample.__closure__
                                for i, cell in enumerate(closure_vars):
                                    if cell.cell_contents and isinstance(cell.cell_contents, nn.Module):
                                        setattr(self, f"{prefix}_downsample_{i}", cell.cell_contents)
                        except:
                            # If we can't extract, just skip
                            print("Warning: Could not register downsample layers from closure.")
                            pass
                    else: 
                        print(f"Warning: Unsupported downsample type for {prefix}: {type(block.downsample)}")
            
            def forward(self, x: Any) -> Any:
                return forward_fn(x)
        
        self._model = Model(self._layer_registry)
        #TODO: Set gpu - do we set by rank?
        if self.gpu_id >= 0:
            if torch.cuda.is_available():
                self._model = self._model.cuda("cuda:{}".format(self.gpu_id))
            else:
                print("Warning: CUDA not available, running on CPU.")
                self._model = self._model.cpu()
        if self.communication:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self._model = DDP(self._model)
        return self._model
    def set_optimizer(self, optimizer, *args, **kwargs):
        assert self._model is not None, "Model must be set before optimizer."
        self._optimizer = optimizer(self._model.parameters(), *args, **kwargs)

    def compute(self, input_data, target) -> None:
        assert self._model is not None
        assert self._optimizer is not None

        self._model.zero_grad()
        if self.gpu_id >= 0 and torch.cuda.is_available():
            input_data = input_data.cuda("cuda:{}".format(self.gpu_id))
            target = target.cuda("cuda:{}".format(self.gpu_id))
        pred = self._model(input_data)

        loss = self._loss_function(pred, target)
        loss.backward()
        # print("weights before update:")
        # for name, param in self._model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.data}")
        self._optimizer.step()
        # print("weights after update:")
        # for name, param in self._model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.data}")
