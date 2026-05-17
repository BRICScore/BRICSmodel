from .feature_data import FeatureData
from .brics_model_wrapper import BRICSModelWrapper
from torch.utils.data import DataLoader
from typing import Optional

class ModelData:
    def __init__(self, feature_data: FeatureData, model_wrapper: BRICSModelWrapper) -> None:
        self.feature_data: FeatureData = feature_data
        self.model_wrapper: BRICSModelWrapper = model_wrapper
        self.train_loader: DataLoader
        self.test_loader: DataLoader
