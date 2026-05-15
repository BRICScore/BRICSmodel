import numpy as np
from torch import Tensor

# this is a basic struct therefore it can be represented with
# a data class of some sort
class FeatureData:
    def __init__(self, data: Tensor, labels: Tensor) -> None:
        self.data = data
        self.labels = labels
        