import sys
from functools import reduce

import pandas as pd
import argparse
import os
import torch
import re
import numpy as np

raw_data_path = "../../data/raw"
features_path = "../../data/features.csv"
labels_path = "../../data/labels.csv"
normalized_path = "../../data/normalized.csv"

feature_cols = {'date': str, 'serial_number': str,
                'smart_197_raw': 'float', 'smart_9_raw': 'float', 'smart_241_raw': 'float', 'smart_187_raw': 'float',
                'failure': 'bool'}

def filter_columns_and_write_back():
    for i, read_filename in enumerate(os.listdir(raw_data_path)):
        print(f"Parsing file {i} out of {len(os.listdir(raw_data_path))}")
        read_filepath = os.path.join(raw_data_path, read_filename)
        df = pd.read_csv(read_filepath)
        features = df[feature_cols.keys()].astype(feature_cols)
        header = not os.path.isfile(features_path)
        features.to_csv(features_path, mode='a', index=None, header=header)

def normalization_and_additional_failure_columns():
    features = pd.read_csv(features_path, parse_dates=['date'])
    print('CSV read finished')

    smart_col_names = ['smart_197_raw', 'smart_9_raw', 'smart_241_raw', 'smart_187_raw']
    smart_means = features[smart_col_names].mean(axis=0)
    smart_stddevs = features[smart_col_names].std(axis=0)

    #replace nan values with column means
    features.fillna(smart_means, inplace=True)

    #normalize to standard normal distribution
    features[smart_col_names] = (features[smart_col_names] - smart_means) / smart_stddevs

    print('Normalization finished')

    #add failure columns
    days_to_look_ahead = 5
    features.rename(columns={'failure': 'fails_within_0_days'}, inplace=True)
    for i in range(1, days_to_look_ahead + 1):
        fails_within_next_i_days = features[['serial_number', 'date', "fails_within_0_days"]].rename(
            columns={"fails_within_0_days": f"fails_within_{i}_days"})
        fails_within_next_i_days.date = fails_within_next_i_days.date - pd.Timedelta(days=i)
        fails_within_next_i_days[f"fails_within_{i}_days"] |= features[f"fails_within_{i-1}_days"]
        features = pd.merge(features, fails_within_next_i_days, on=['serial_number', 'date'])
        print(f"Lookahead day {i} merged")

    features.to_csv(normalized_path, index=None)
    print('Final features file written')

if __name__ == "__main__":
    filter_columns_and_write_back()
    normalization_and_additional_failure_columns()