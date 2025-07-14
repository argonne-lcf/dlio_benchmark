

from dlio_benchmark.common.enumerations import FrameworkType

from abc import ABC, abstractmethod
from typing import Any, Tuple, Type

from dlio_benchmark.model_factory.tf_layer import TensorFlowLayers
from dlio_benchmark.model_factory.torch_layer import PyTorchLayers



class UnifiedModel(ABC):
    """Abstract base class for unified models"""
    
    def __init__(self, framework: FrameworkType):
        self.framework = framework
        self.layers = []
        # Initialize the appropriate layer factory
        if framework == FrameworkType.PYTORCH:
            self.layer_factory = PyTorchLayers
        elif framework == FrameworkType.TENSORFLOW:
            self.layer_factory = TensorFlowLayers
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    @abstractmethod
    def build_model(self) -> None:
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass through the model"""
        pass


