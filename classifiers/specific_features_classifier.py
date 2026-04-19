import itertools
from typing import Optional
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
import json
import os

FEATURES_PATH = './data/features'
CLEAN_PATH = '/data/clean'
RAW_PATH = '/data/raw'

class FeatureData():
    def __init__(self):
        self.feature_files = []
        self.feature_colors = []
        self.features: np.ndarray = np.array([])
        self.feature_count = None
        self.features_scaled: np.ndarray = np.array([])

        self.feature_index = 0
        self.feature_keys = {}
        self.person_colors = {} # dictionary for colors of data points for different person labels
        self.person_indices = {} # dictionary holding arrays of indices in feature data for people
        self.person_initials = [] # array holding all initials for labels in legend

def parse_features_line(line, feature_data, filename):
    feature_vector = []
    temp = filename.split('_') # TODO
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
    create_indices_for_features(feature_data, last_filename)
    npfeatures = np.array(features)
    feature_data.features_scaled = StandardScaler().fit_transform(npfeatures)
    return np.array(npfeatures)

def create_indices_for_features(feature_data, filename):
    record = None
    with open(FEATURES_PATH+'/'+filename, "r") as file:
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

def establish_best_features(feature_data: FeatureData, cov: np.ndarray) -> tuple[tuple[int, int], float]:
    cov_matrix = np.cov(feature_data.features_scaled, rowvar=False)
    n = feature_data.features.shape[1]
    best_pair = (-1, -1)
    best_value = -np.inf

    for i in range(n):
        for j in range(i + 1, n):
            val = cov[i, i] + cov[j, j] + 2 * cov[i, j]
            if val > best_value:
                best_value = val
                best_pair = (i, j)

    return best_pair, best_value

def SVM_classification(feature_data: FeatureData, best_pair: tuple[int, int]):
    one, two = feature_data.feature_keys[best_pair[0]], feature_data.feature_keys[best_pair[1]]
    print("Best features are:", f"{one} and {two}")
    X = feature_data.features[:,np.array([best_pair[0], best_pair[1]])]
    print(X)
    for pair in list(itertools.combinations(feature_data.person_initials, 2)):
        person1, person2 = pair
        records1 = X[feature_data.person_indices[person1]]
        labels1 = [person1 for record in records1]
        records2 = X[feature_data.person_indices[person2]]
        labels2 = [person2 for record in records2]
        # columns are the 2 reduced features
        cut_x = np.concatenate((records1, records2), axis=0)
        y = np.concatenate((labels1, labels2), axis=0)

        svm = SVC(kernel="linear", C=1)
        svm.fit(cut_x,y)

        DecisionBoundaryDisplay.from_estimator(
            svm,
            cut_x,
            response_method="predict",
            alpha=0.8,
            cmap="Pastel1",
            xlabel="x",
            ylabel="y"
        )

        W=svm.coef_[0]
        I=svm.intercept_
        a = -W[0]/W[1]
        b = I[0]/W[1]

        line_x = np.linspace(np.min(cut_x[:,0]), np.max(cut_x[:,0]), num=10)
        line_y = [a*x - b for x in line_x]
        plt.plot(line_x, line_y, "--", c="k")
        x_pad = (np.max(cut_x[:,0]) - np.min(cut_x[:,0])) * 0.05
        y_pad = (np.max(cut_x[:,1]) - np.min(cut_x[:,1])) * 0.05

        plt.xlim(np.min(cut_x[:,0]) - x_pad, np.max(cut_x[:,0]) + x_pad)
        plt.ylim(np.min(cut_x[:,1]) - y_pad, np.max(cut_x[:,1]) + y_pad)

        y_pred = svm.predict(cut_x)
        length_y = len(y_pred)
        counter = 0
        for i in range(length_y):
            if y[i] == y_pred[i]:
                counter += 1
        
        print(f"Accuracy for {person1} and {person2}: {(counter/length_y)*100:.2f}%")

        plt.title(f"{person1} and {person2} divided by linear SVM")
        plt.scatter(records1[:, 0], records1[:, 1], c="red")
        plt.scatter(records2[:, 0], records2[:, 1], c="blue")
        plt.legend(["Dividing line", person1, person2])
        plt.xlabel(one)
        plt.ylabel(two)
        plt.show()

def specific_features_classifier():
    feature_data = FeatureData()

    feature_data.features = feature_loading(feature_data)
    # print(feature_data.features)
    # print(feature_data.features_scaled)
    # print(feature_data.features.shape)
    cov_matrix = np.cov(feature_data.features_scaled, rowvar=False)
    best_pair, best_value = establish_best_features(feature_data=feature_data, cov=cov_matrix)
    SVM_classification(feature_data=feature_data, best_pair=best_pair)

def main():
    specific_features_classifier()

if __name__ == "__main__":
    main()  