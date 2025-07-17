 


from functools import partial
from dlio_benchmark.common.enumerations import FrameworkType, Loss
from dlio_benchmark.model.layer import LayerFactoryBase
from dlio_benchmark.model.model import UnifiedModel
from typing import Any, Optional, Type, Union

#TODO: Verify correctness of resnet50

class ResNet50Block:
    """ResNet50 basic block"""

    def __init__(self, layer_factory, framework: FrameworkType, in_channels: int,
                 out_channels: int, stride: int = 1, downsample: Optional[Any] = None):
        self.framework = framework
        self.conv1 = layer_factory.conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = layer_factory.batch_norm(out_channels)
        self.conv2 = layer_factory.conv2d(out_channels, out_channels, 3, 
                                         stride=stride, padding=1, bias=False)
        self.bn2 = layer_factory.batch_norm(out_channels)
        self.conv3 = layer_factory.conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = layer_factory.batch_norm(out_channels * 4)
        self.relu = layer_factory.relu()
        self.downsample = downsample
    
    def __call__(self, x: Any) -> Any:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # Element-wise addition
        if self.framework == FrameworkType.PYTORCH:
            out += residual
        else:  # TensorFlow
            import tensorflow as tf
            out = tf.add(out, residual)
        
        out = self.relu(out)
        return out

class ResNet50(UnifiedModel):
    """ResNet50 implementation"""

    def __init__(self, framework: FrameworkType,num_classes: int = 1000):
        super().__init__(framework, Loss.CE)
        self.num_classes = num_classes
        self.build_model()
        # bound_forward = partial(self.forward, self)
        self._model = self.layer_factory.get_model(self.forward)
        if framework == FrameworkType.PYTORCH:
            # TODO Make configurable
            import torch
            self.layer_factory.set_optimizer(torch.optim.SGD, 1,
                                momentum=1,
                                weight_decay=1)
        else: 
            import tensorflow as tf
            self.layer_factory.set_optimizer(tf.optimizers.SGD, 0.1)

    def build_model(self):
        """Build ResNet50 architecture"""
        # Initial convolution
        self.conv1 = self.layer_factory.conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = self.layer_factory.batch_norm(64)
        self.relu = self.layer_factory.relu()
        self.maxpool = self.layer_factory.max_pool2d(3, stride=2)
        
        # ResNet blocks (simplified - just showing the concept)
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
        # Final layers
        self.avgpool = self.layer_factory.adaptive_avg_pool2d((1, 1))
        self.flatten = self.layer_factory.flatten()
        self.fc = self.layer_factory.linear(2048, self.num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """Create a ResNet layer with multiple blocks"""
        layers = []
        downsample = None
        
        if stride != 1 or in_channels != out_channels * 4:
            downsample_conv = self.layer_factory.conv2d(in_channels, out_channels * 4, 1, 
                                                        stride=stride, bias=False)
            downsample_bn = self.layer_factory.batch_norm(out_channels * 4)
            # In practice, you'd need to sequence these layers properly
            downsample = lambda x: downsample_bn(downsample_conv(x))
        
        layers.append(ResNet50Block(self.layer_factory, self.framework, in_channels, 
                                    out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(ResNet50Block(self.layer_factory, self.framework, 
                                        out_channels * 4, out_channels))
        
        return layers

    def forward(self, x: Any) -> Any:
        """Forward pass through ResNet50"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Apply ResNet blocks
        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x
    