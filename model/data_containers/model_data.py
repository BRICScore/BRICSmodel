from .feature_data import FeatureData
from .brics_model_wrapper import BRICSModelWrapper
from torch.utils.data import DataLoader
from typing import Optional

class ModelData:
    def __init__(self, feature_data: FeatureData) -> None:
        self.feature_data = feature_data
        self.model_wrapper: Optional[BRICSModelWrapper]
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        self.__setup_loaders()

    def __setup_loaders(self): #TODO after parser scripts
        pass

    def scale_data(self): #TODO after parser scripts
        pass