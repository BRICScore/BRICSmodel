from model.data_containers import *
from .parse_features import parse_features_from_data

def train_model():
    model_data = parse_features_from_data()
    print(model_data.feature_data.data)
    print(model_data.feature_data.data.size())
    print(model_data.feature_data.labels)
    print(model_data.feature_data.labels.size())
