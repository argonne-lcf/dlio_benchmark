"""Exercise shared batch preparation with each backend's real conversions."""

import numpy as np
import pytest


@pytest.fixture(params=["torch", "tensorflow"])
def framework(request):
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from dlio_benchmark.framework.torch_framework import TorchFramework

        instance = TorchFramework.__new__(TorchFramework)
        instance.device = torch.device("cpu")
        return instance, torch.float32, torch.int64

    tf = pytest.importorskip("tensorflow")
    from dlio_benchmark.framework.tf_framework import TFFramework

    return TFFramework.__new__(TFFramework), tf.float32, tf.int32


@pytest.mark.parametrize("layout", ["tuple", "list", "nested", "data", "image", "input", "inputs"])
def test_prepare_batch_preserves_values_and_sets_backend_dtypes(framework, layout):
    instance, input_dtype, target_dtype = framework
    inputs = np.arange(6, dtype=np.uint8).reshape(2, 3)
    target = np.array([0, 1], dtype=np.int64)
    if layout == "tuple":
        batch = (inputs, target)
    elif layout == "list":
        batch = [inputs, target]
    elif layout == "nested":
        batch = [((inputs, target),)]
    else:
        target_key = dict(data="label", image="labels", input="target", inputs="targets")[layout]
        batch = {layout: inputs, target_key: target}

    prepared_inputs, prepared_target = instance._prepare_batch(batch)

    np.testing.assert_array_equal(prepared_inputs.numpy(), inputs)
    np.testing.assert_array_equal(prepared_target.numpy(), target)
    assert prepared_inputs.dtype == input_dtype
    assert prepared_target.dtype == target_dtype
    if hasattr(instance, "device"):
        assert prepared_inputs.device == prepared_target.device == instance.device


@pytest.mark.parametrize("layout", ["bare", "wrapped", "dict", "single_key"])
def test_prepare_batch_rejects_missing_target(framework, layout):
    instance, _, _ = framework
    inputs = np.ones((2, 3))
    batch = {
        "bare": inputs,
        "wrapped": [inputs],
        "dict": {"inputs": inputs},
        "single_key": {"custom": inputs},
    }[layout]

    with pytest.raises(ValueError, match="Native model training requires a target"):
        instance._prepare_batch(batch)


def test_prepare_batch_rejects_unrecognized_dictionary(framework):
    instance, _, _ = framework

    with pytest.raises(ValueError, match="Could not find model input"):
        instance._prepare_batch({"unknown": np.zeros(2), "labels": np.zeros(2)})


def test_as_tensor_preserves_native_tensor_identity(framework):
    instance, _, _ = framework
    tensor = instance._as_tensor(np.arange(3, dtype=np.float32))

    assert instance._as_tensor(tensor) is tensor


class ArrayValue:
    def __init__(self, value):
        self.value = value

    def as_array(self):
        return self.value


class TensorListValue:
    def __init__(self, value):
        self.value = value

    def as_tensor(self):
        return ArrayValue(self.value)


@pytest.mark.parametrize("wrapper", [ArrayValue, TensorListValue])
def test_prepare_batch_unwraps_array_and_tensor_list_values(framework, wrapper):
    instance, _, _ = framework
    inputs = np.arange(6, dtype=np.float32).reshape(2, 3)
    target = np.array([0, 1])

    prepared_inputs, prepared_target = instance._prepare_batch(
        (wrapper(inputs), wrapper(target))
    )

    np.testing.assert_array_equal(prepared_inputs.numpy(), inputs)
    np.testing.assert_array_equal(prepared_target.numpy(), target)


class DLPackValue:
    def __init__(self, value):
        self.value = value
        self.consumed = False

    def __dlpack__(self, stream=None):
        self.consumed = True
        return self.value.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self.value.__dlpack_device__()


def test_as_tensor_uses_backend_dlpack_conversion(framework):
    instance, _, _ = framework
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    value = DLPackValue(array)

    tensor = instance._as_tensor(value)

    assert value.consumed
    np.testing.assert_array_equal(tensor.numpy(), array)


def test_tensorflow_accepts_torch_tensors_requiring_gradients():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorflow")
    from dlio_benchmark.framework.tf_framework import TFFramework

    inputs = torch.ones((2, 3), requires_grad=True)
    target = torch.tensor([0, 1])
    instance = TFFramework.__new__(TFFramework)

    prepared_inputs, prepared_target = instance._prepare_batch((inputs, target))

    np.testing.assert_array_equal(prepared_inputs.numpy(), inputs.detach().numpy())
    np.testing.assert_array_equal(prepared_target.numpy(), target.numpy())
