import argparse
import random
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional

from classifiers.specific_features_classifier import FeatureData
from utils.measurement_dataset_control import MeasurementDatasetHook
from utils.file_processing.measurement_data_builder import MeasurementDataBuilder
from data_containers import MeasurementData

NO_OF_FEATURES_AFTER_ALG = 2
FEATURES_PATH = './features'

def parse_features_line(line, feature_data, filename):
    feature_vector = []
    temp = filename.split('_')
    person = temp[2][:2]
    for key in line:
        if isinstance(line[key], list):
            for val in line[key]:
                feature_vector.append(val)
        elif isinstance(line[key], str):
            # print(f"{line[key]}")
            person = line[key]
        else:
            feature_vector.append(line[key])
    if person not in feature_data.person_initials:
        feature_data.person_initials.append(person)
        color = random.randrange(0, 2**24)
        hex_color = hex(color)
        color_part = hex_color[2:]
        while len(color_part) < 6:
                color_part = "0" + color_part
        rand_color = "#" + color_part
        feature_data.person_colors[person] = rand_color
        feature_data.person_indices[person] = []
    feature_data.person_indices[person].append(feature_data.feature_index)
    feature_data.feature_index += 1
    return feature_vector

def create_indices_for_features(feature_data, filepath):
    record = None
    with open(filepath, "r") as file:
        record = file.readline() # skip metadata - works with files without metadata if the file is 2 segments long
        record = file.readline()
    i = 0
    json_line = json.loads(record)
    for key in json_line:
        if isinstance(json_line[key], list):
            index = 1
            for val in json_line[key]:
                feature_data.feature_keys[i] = key + str(index)
                index += 1
                i += 1
        else:
            feature_data.feature_keys[i] = key
            i += 1

def feature_loading(feature_data):
    features = []
    last_filename: str = 'extracted_features.jsonl'
    with os.scandir(FEATURES_PATH) as es:
        for e in es:
            print(e.name)
            if e.is_file() and e.name.endswith('.jsonl'):
                last_filename = e.name
                with open(e.path, encoding='utf-8') as file:
                    record = file.readline()
                    while record:
                        json_record = json.loads(record)
                        feature_vector = parse_features_line(json_record, feature_data, e.name)
                        features.append(feature_vector)
                        record = file.readline()
    feature_data.feature_count = len(features[0])
    feature_data.person_colors = {"JD": "orange", "MJ": "green", "MK": "blue", "DS": "red"} ###########TODO############
    create_indices_for_features(feature_data, FEATURES_PATH+'/'+last_filename)
    return np.array(features)

def load_features_from_database_zip(feature_data: FeatureData) -> np.ndarray:
    combined_features: np.ndarray = np.array([])
    hook = MeasurementDatasetHook(target="features")
    measurement_data = MeasurementData()
    measurement_data_builder = MeasurementDataBuilder(measurement_data_container=measurement_data)
    final_filepath: Optional[Path] = None

    first_go = True
    for filepath in hook:
        measurement_data_builder.build_data(filepath=filepath, target="features")
        current_file_features = []
        length = 0
        for key, value in measurement_data.data_features.__dict__.items():
            length = len(value)
            if value.ndim == 1:
                current_file_features.append(value)
            elif value.ndim == 2:
                for i in range(value.shape[1]): #for every column
                    current_file_features.append(value[:,i])
        if first_go:
            first_go = False
            # vstack requires uniform dimensions across every dimensions except the one stacked
            # so in order for the loop to work we need to define an empty row that we omit later in return
            combined_features = np.empty(shape=(1,len(current_file_features)))
        combined_features = np.vstack((combined_features, np.array(current_file_features).T))

        # person = "test"
        person = measurement_data.metadata.labels.person_data.person_id

        if person not in feature_data.person_initials:
            feature_data.person_initials.append(person)
            color = random.randrange(0, 2**24)
            hex_color = hex(color)
            color_part = hex_color[2:]
            while len(color_part) < 6:
                    color_part = "0" + color_part
            rand_color = "#" + color_part
            feature_data.person_colors[person] = rand_color
            feature_data.person_indices[person] = []
        for index in range(length):
            feature_data.person_indices[person].append(feature_data.feature_index + index)
        feature_data.feature_index += length
        final_filepath = filepath

    if final_filepath:
        create_indices_for_features(feature_data=feature_data, filepath=final_filepath)

    return np.array(combined_features[1:,:]) #skip the empty row

def parser_setup():
    parser = argparse.ArgumentParser(description="Visualization work mode information")

    parser.add_argument('--local', action='store_true',
                    help='A boolean switch for local files instead of files from database zip')
    return parser


def main():
    parser = parser_setup()
    args = parser.parse_args()

    feature_data = FeatureData()
    feature_data.features = load_features_from_database_zip(feature_data=feature_data)

if __name__ == "__main__":
    main()