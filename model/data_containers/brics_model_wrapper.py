from torch import nn
from pathlib import Path
from typing import Optional

class BRICSModelWrapper:
    def __init__(self) -> None:
        self.model: Optional[nn.Module] = None
        self.parameters: Optional[dict]
        self.people_keys: Optional[dict]
        self.feature_keys: Optional[dict]
    
    def get_person(self, label: int) -> Optional[str]:
        if self.people_keys:
            return self.people_keys[label]
        else:
            return None
    
    def identify_person(self, feedData): #TODO during UI integration
        pass

    def save_model(self, filepath: Path): #TODO during model training
        pass

    def load_model(self, filepath: Path = Path("../model/model.pt")): #TODO during model training
        pass