from typing import Any

from torch import nn

class BRICSModel(nn.Module):
    def __init__(self, feature_count: int, n_classes: int) -> None:
        super().__init__()

        self.input_size = feature_count
        self.output_size = n_classes

        self.network = nn.Sequential(
                nn.Linear(feature_count, 32),
                nn.ReLU(),
                nn.Linear(32,64),
                nn.ReLU(),
                nn.Linear(64, n_classes)
            )
        
        def forward(self, x):
            return self.network(x)