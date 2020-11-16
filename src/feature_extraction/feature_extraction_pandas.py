import pandas as pd
import argparse
import os
import torch
import re
import numpy as np

raw_data_path = "../../data/raw"
features_path = "../../data/features.h5"
labels_path = "../../data/labels.h5"

smart_values = ['smart_197_raw', 'smart_9_raw', 'smart_241_raw', 'smart_187_raw']

def feature_extract(df):
    return df[smart_values].astype('float'), df['failure']

def write_feature_columns_to_hdf():
    for i, read_filename in enumerate(os.listdir(raw_data_path)):
        print(f"Parsing file {i} out of {len(os.listdir(raw_data_path))}")
        read_filepath = os.path.join(raw_data_path, read_filename)
        df = pd.read_csv(read_filepath)
        X, y = feature_extract(df)
        X.to_hdf(features_path, key='df', append=True, mode='a')
        y.to_hdf(labels_path, key='df', append=True, mode='a')

def normalize_hdf_columns():
    features = pd.read_hdf(features_path)

    col_means = features.mean(axis=0)
    col_stddevs = features.std(axis=0)

    #replace nan values with column means
    features.fillna(col_means, inplace=True)

    #normalize to standard normal distribution
    features = (features - col_means) / col_stddevs

    features.to_hdf(features_path, key='df', mode='a', append=True)

if __name__ == "__main__":
    write_feature_columns_to_hdf()
    normalize_hdf_columns()