from dlio_benchmark.common.enumerations import FrameworkType, Model
from dlio_benchmark.model_factory.model import UnifiedModel
from dlio_benchmark.model_factory.models.resnet import ResNet50


class ModelFactory:
    @staticmethod
    def create_model(framework: FrameworkType, model_type: Model) -> UnifiedModel:
        if model_type == Model.RESNET:
            return ResNet50(framework)
        raise ValueError(f"Unsupported model type: {model_type}")