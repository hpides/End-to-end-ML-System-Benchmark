import sys

from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import time
import json

raw_data_path = "data/raw"
features_path = "data/features.csv"
labels_path = "data/labels.csv"
normalized_path = "data/normalized.csv"

feature_cols = {'date': str, 'serial_number': str,
                'smart_197_raw': 'float', 'smart_9_raw': 'float', 'smart_241_raw': 'float', 'smart_187_raw': 'float',
                'failure': 'bool'}

time_measuring_data = dict()

def filter_columns_and_write_back():
    start = time.time()
    read_files = os.listdir(raw_data_path)
    with tqdm(total=len(read_files)) as progress_bar:
        for read_filename in read_files:
            progress_bar.update()
            progress_bar.set_description(f"Parsing raw file {progress_bar.n}/{progress_bar.total}: {read_filename}")
            read_filepath = os.path.join(raw_data_path, read_filename)
            df = pd.read_csv(read_filepath)
            features = df[feature_cols.keys()].astype(feature_cols)
            header = not os.path.isfile(features_path)
            features.to_csv(features_path, mode='a', index=None, header=header)
    end = time.time()
    time_measuring_data['time_parse_raw_files'] = end - start

def add_days_to_failure_column_to_group(group):
    if group.failure.any():
        group['days_to_failure'] = (group.date.max() - group.date).dt.days
    else:
        group['days_to_failure'] = -1
    return group

def normalize_and_add_days_to_failure_column():
    print('Reading csv for normalization')
    start = time.time()
    features = pd.read_csv(features_path, parse_dates=['date'])
    print(sys.getsizeof(features))
    end = time.time()
    time_measuring_data['time_read_parsed_file'] = end - start
    print('CSV read for normalization finished')

    start = time.time()
    smart_col_names = ['smart_197_raw', 'smart_9_raw', 'smart_241_raw', 'smart_187_raw']
    smart_means = features[smart_col_names].mean(axis=0)
    smart_stddevs = features[smart_col_names].std(axis=0)

    #replace nan values with column means
    features.fillna(smart_means, inplace=True)

    #normalize to standard normal distribution
    features[smart_col_names] = (features[smart_col_names] - smart_means) / smart_stddevs

    print('Normalization finished')
    end = time.time()
    time_measuring_data['time_normalization'] = end - start

    #add day-to-failure columns
    start = time.time()
    print('Adding days_to_failure column')
    tqdm.pandas()
    features = features.groupby('serial_number').progress_apply(add_days_to_failure_column_to_group)
    end = time.time()
    time_measuring_data['time_add_day_to_failure_col'] = end - start

    print('Final csv write')
    features.to_csv(normalized_path, index=None)


if __name__ == "__main__":
    #filter_columns_and_write_back()
    normalize_and_add_days_to_failure_column()

    print(time_measuring_data)