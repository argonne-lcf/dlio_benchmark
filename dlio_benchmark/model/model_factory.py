"""Construct native framework models without importing unused backends."""

from dlio_benchmark.common.enumerations import FrameworkType, Model


class ModelFactory:
    @staticmethod
    def create_model(framework: FrameworkType, model_type: Model):
        """Return a raw ``torch.nn.Module`` or ``tf.keras.Model``.

        Device placement, distributed wrapping, optimization, and training are
        framework concerns and intentionally do not belong to this factory.
        """
        if model_type in (Model.SLEEP, Model.DEFAULT):
            return None

        if framework not in (FrameworkType.PYTORCH, FrameworkType.TENSORFLOW):
            raise ValueError(f"Unsupported framework: {framework}")
        raise ValueError(f"Unsupported model type: {model_type}")
