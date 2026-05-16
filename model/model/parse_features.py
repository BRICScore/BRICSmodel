import json
from model.data_containers import *
import os
from pathlib import Path
from torch import Tensor, tensor
import torch
import numpy as np

def create_indices_for_features(record: str) -> dict:
    feature_keys = {}

    i = 0
    json_line = json.loads(record)
    for key in json_line:
        if isinstance(json_line[key], list):
            index = 1
            for val in json_line[key]:
                feature_keys[i] = key + str(index)
                index += 1
                i += 1
        else:
            feature_keys[i] = key
            i += 1
    return feature_keys

def parse_record(record):
    features = json.loads(record)
    feature_vector = []
    for key, val in features.items():
        if type(val) == list:
            for number in val:
                feature_vector.append(number)
        else:
            feature_vector.append(val)
    return feature_vector

def parse_features_from_data() -> ModelData:
    model_wrapper = BRICSModelWrapper()
    people_keys = {}
    feature_keys = {}
    first_go = True
    row_index = 0
    person_index = 0
    data = []
    labels = []
    with os.scandir(Path("./data/features")) as files:
        for file in files:
            if first_go:
                with open(file, "r") as f:
                    f.readline()
                    feature_keys = create_indices_for_features(f.readline())
            with open(file, "r") as f:
                counter = 0
                metadata = json.loads(f.readline())
                record = f.readline()
                while record:
                    labels.append([person_index])
                    feature_vector = parse_record(record)
                    data.append(feature_vector)
                    record = f.readline()
            person_index += 1
    np_data = np.array(data)
    np_labels = np.array(labels)
    feature_data = FeatureData(data=tensor(np_data, dtype=torch.float32), labels=tensor(np_labels, dtype=torch.int))
    model_data = ModelData(feature_data=feature_data)
    model_data.model_wrapper = model_wrapper
    return model_data

