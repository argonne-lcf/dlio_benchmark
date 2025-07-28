


from typing import Any
from dlio_benchmark.common.enumerations import Loss


def torch_loss(loss: Loss) -> Any:
    """Convert a Loss enumeration to a PyTorch loss function."""
    import torch.nn as nn
    if loss == Loss.MSE:
        return nn.MSELoss()
    elif loss == Loss.CE:
        return nn.CrossEntropyLoss()
    elif loss == Loss.NONE:
        return None
    else:
        raise ValueError(f"Unsupported loss function: {loss}")
    

def tf_loss(loss: Loss) -> Any:
    import tensorflow.keras.losses as losses # type: ignore
    if loss == Loss.MSE:
        return losses.MeanSquaredError()
    elif loss == Loss.CE:
        return losses.CategoricalCrossentropy()
    elif loss == Loss.NONE:
        return None
    else:
        raise ValueError(f"Unsupported loss function: {loss}")
