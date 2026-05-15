import argparse
import random
import os
import json
import dotenv
import numpy as np
from pathlib import Path
from typing import Optional

from classifiers.specific_features_classifier import FeatureData
from utils.measurement_dataset_control import MeasurementDatasetHook
from utils.file_processing.measurement_data_builder import MeasurementDataBuilder
from data_containers import MeasurementData, MeasurementMetadata
import sys
sys.path.append("utils")
from config import *
from database_access.database_handler import *


NO_OF_FEATURES_AFTER_ALG = 2
FEATURES_PATH = './data/features'

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

def load_features_from_database_zip(feature_data: FeatureData) -> tuple[np.ndarray, list]:
    combined_features: np.ndarray = np.array([])
    hook = MeasurementDatasetHook(target="features")
    measurement_data = MeasurementData()
    measurement_data_builder = MeasurementDataBuilder(measurement_data_container=measurement_data)
    metadata_rows = []
    final_filepath: Optional[Path] = None

    first_go = True
    for filepath in hook:
        measurement_data_builder.build_data(filepath=filepath, target="features")
        metadata_rows.append({
            "person_id": measurement_data.metadata.labels.person_data.person_id,
            "condition": measurement_data.metadata.labels.person_data.condition,
            "activity": measurement_data.metadata.labels.activity,
            "timestamp": measurement_data.metadata.timestamp,
            "duration_ms": measurement_data.metadata.duration_ms,
            "filepath_raw": str(measurement_data.metadata.filepath_raw),
            "filepath_clean": str(measurement_data.metadata.filepath_clean),
            "filepath_features": str(measurement_data.metadata.filepath_features)
        })
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

    return np.array(combined_features[1:,:]), metadata_rows  #skip the empty row

import json
import re
from pathlib import Path

def get_feature_data(feature_data: FeatureData, metadata_rows: list):
    feature_data_folder = Path(FEATURES_PATH)
    feature_data_folder.mkdir(parents=True, exist_ok=True)

    for row_index, row in enumerate(feature_data.features):
        metadata = {}
        # TODO: this is very temporary, for now db returns 3 different files per querry (for my id)
        # and so i have to choose which metadata to use
        # should there be only one file per querry or the 3 is normal?
        metadata["person_id"] = metadata_rows[0]["person_id"]
        metadata["condition"] = metadata_rows[0]["condition"]
        metadata["activity"] = metadata_rows[0]["activity"]

        record = {}
        for feature_index, feature_name in feature_data.feature_keys.items():
            value = row[feature_index]
            if hasattr(value, "item"):
                value = value.item()
            record[feature_name] = value

        out_path = feature_data_folder / f"features_{metadata['person_id']}_{row_index}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")
            f.write(json.dumps(record) + "\n")

def parser_setup():
    parser = argparse.ArgumentParser(description="Visualization work mode information")

    parser.add_argument('--local', action='store_true',
                    help='A boolean switch for local files instead of files from database zip')
    return parser


def main():
    parser = parser_setup()
    args = parser.parse_args()

    dotenv.load_dotenv()
    dbh = DatabaseHandler()
    dbh.downloadMeasurement()

    feature_data = FeatureData()
    feature_data.features, metadata_rows = load_features_from_database_zip(feature_data=feature_data)
    # print(feature_data.features)

    get_feature_data(feature_data=feature_data, metadata_rows=metadata_rows)



if __name__ == "__main__":
    main()