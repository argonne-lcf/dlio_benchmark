from dlio_benchmark.common.enumerations import FrameworkType, Model
from dlio_benchmark.model.model import UnifiedModel
from dlio_benchmark.model.impl.resnet import ResNet50


class ModelFactory:
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_model(framework: FrameworkType, model_type: Model, communication: bool = False, gpu_id: int = -1) -> UnifiedModel:
        if model_type == Model.RESNET:
            return ResNet50(framework, communication, gpu_id)
        elif model_type in (Model.SLEEP, Model.DEFAULT):
            return None
        raise ValueError(f"Unsupported model type: {model_type}")