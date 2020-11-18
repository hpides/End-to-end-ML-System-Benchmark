from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import time
import json

raw_data_path = "../../data/raw"
features_path = "../../data/features.csv"
labels_path = "../../data/labels.csv"
normalized_path = "../../data/normalized.csv"

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

def add_lookahead_failure_cols_to_group(group, num_look_ahead_days):
    copy = group.copy()
    entry_count = len(copy)

    if copy.failure.any():
        copy.sort_values('date', inplace=True)
        ones = np.ones((entry_count, num_look_ahead_days))
        failure_values_float = np.fliplr(np.tril(ones, num_look_ahead_days - entry_count))
        failure_values_bool = failure_values_float != 0
    else:
        failure_values_bool = np.zeros((entry_count, num_look_ahead_days)) != 0

    copy.drop(columns=['failure'], inplace=True)
    col_names = [f"fails_within_{i}_days" for i in range(1, num_look_ahead_days + 1)]
    failure_features = pd.DataFrame(data=failure_values_bool, columns=col_names)
    failure_features.index = copy.index

    return pd.concat([copy, failure_features], axis=1)

def normalization_and_additional_failure_columns():
    print('Reading csv for normalization')
    start = time.time()
    features = pd.read_csv(features_path, parse_dates=['date'])
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

    #add failure columns
    start = time.time()
    num_look_ahead_days = 5
    num_serial_numbers = features.serial_number.nunique()
    grouped_by_serial_number = features.groupby('serial_number')

    with tqdm(total=num_serial_numbers, desc='Adding lookahead failure columns') as progress_bar:
        for serial_number, group in grouped_by_serial_number:
            group_with_failure_cols = add_lookahead_failure_cols_to_group(group, num_look_ahead_days)
            header = not os.path.isfile(normalized_path)
            group_with_failure_cols.to_csv(normalized_path, mode='a', index=None, header=header)
            progress_bar.update()

    end = time.time()
    time_measuring_data['time_add_lookahead_failure_cols'] = end - start


if __name__ == "__main__":
    filter_columns_and_write_back()
    normalization_and_additional_failure_columns()

    print(time_measuring_data)