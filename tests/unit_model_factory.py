from dlio_benchmark.common.enumerations import FrameworkType, Model
import pytest

import torch
import tensorflow as tf

from dlio_benchmark.model import ModelFactory

params = [
    (FrameworkType.PYTORCH, Model.RESNET, (1, 3, 224, 224), torch.Tensor),
    (FrameworkType.TENSORFLOW, Model.RESNET, (1, 224, 224, 3), tf.Tensor),
    (FrameworkType.PYTORCH, Model.RESNET, (1, 3, 64, 64), torch.Tensor),  # Increased size from 32x32 to 64x64
    (FrameworkType.TENSORFLOW, Model.RESNET, (1, 64, 64, 3), tf.Tensor),  # Increased size from 32x32 to 64x64
]

@pytest.mark.parametrize("framework, model_type, input_shape, expected_type", params)
def test_model_factory_output_type(framework, model_type, input_shape, expected_type):
    model = ModelFactory().create_model(framework, model_type, communication=False, gpu_id=0)
    if framework == FrameworkType.PYTORCH:
        input_data = torch.randn(*input_shape)
    else:
        input_data = tf.random.normal(input_shape)
    # Backward pass
    random_out = torch.randn(1, 1000) if framework == FrameworkType.PYTORCH else tf.random.normal((1, 1000))
    assert random_out.shape == (1, 1000)
    model.compute([input_data, random_out])


if __name__ == "__main__":
    # Run test_model_factory
    for param in params:
        test_model_factory_output_type(*param)