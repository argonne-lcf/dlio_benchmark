

from dlio_benchmark.common.enumerations import FrameworkType, Loss

from abc import ABC, abstractmethod
from typing import Any, Tuple, Type

from dlio_benchmark.model.loss_fn import tf_loss, torch_loss
from dlio_benchmark.model.tf_layer import TensorFlowLayers
from dlio_benchmark.model.torch_layer import PyTorchLayers



class UnifiedModel(ABC):
    """Abstract base class for unified models"""

    def __init__(self, framework: FrameworkType, loss_function: Loss, communication: bool = False):
        self.framework = framework
        self.layers = []
        self._model = None
        # Initialize the appropriate layer factory
        if framework == FrameworkType.PYTORCH:
            self.layer_factory = PyTorchLayers(torch_loss(loss_function), communication)
        elif framework == FrameworkType.TENSORFLOW:
            self.layer_factory = TensorFlowLayers(tf_loss(loss_function), communication)
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


    def compute(self, batch):
        self.layer_factory.compute(batch)

class TorchModel(UnifiedModel):
    """Torch implementation of the unified model"""

    def __init__(self):
        super().__init__(FrameworkType.PYTORCH)
        

    
