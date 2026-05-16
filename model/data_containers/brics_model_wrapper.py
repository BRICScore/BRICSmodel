import torch
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
        # firstly, save the self.parameters to "parameters.jsonl" + input/output sizes
        # torch.save(model, PATH)
        pass

    def load_model(self, filepath: Path = Path("../model/model.pt")): #TODO during model training
        # load input/output sizes and parameters from a separate file called "parameters.json"
        # (try & except for the model input and output size) 
        # (in case number of features or number of people differs in saved model)

        # Model class must be defined somewhere
        # model = torch.load(PATH, weights_only=False)
        # model.eval()
        pass