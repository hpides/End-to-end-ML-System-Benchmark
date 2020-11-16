import pandas as pd
import argparse
import os
import torch
import re
import numpy as np

raw_data_path = "../../data/raw"
features_path = "../../data/features.csv"
labels_path = "../../data/labels.csv"

feature_cols = {'date': str, 'serial_number': str,
                'smart_197_raw': 'float', 'smart_9_raw': 'float', 'smart_241_raw': 'float', 'smart_187_raw': 'float'}

def feature_extract(df):
    return df[feature_cols.keys()].astype(feature_cols), df['failure']

def write_feature_columns_to_csv():
    for i, read_filename in enumerate(os.listdir(raw_data_path)):
        print(f"Parsing file {i} out of {len(os.listdir(raw_data_path))}")
        read_filepath = os.path.join(raw_data_path, read_filename)
        df = pd.read_csv(read_filepath)
        X, y = feature_extract(df)
        header = not os.path.isfile(features_path)
        X.to_csv(features_path, mode='a', index=None, header=header)
        y.to_csv(labels_path, mode='a', index=None, header=header)

def normalization():
    features = pd.read_csv(features_path)

    col_means = features.mean(axis=0)
    col_stddevs = features.std(axis=0)

    #replace nan values with column means
    features.fillna(col_means, inplace=True)

    #normalize to standard normal distribution
    features = (features - col_means) / col_stddevs

    features.to_csv(features_path, mode='a', append=True)

if __name__ == "__main__":
    write_feature_columns_to_csv()
    pass