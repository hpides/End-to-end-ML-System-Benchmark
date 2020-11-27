import sys

import h5py
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
hdf5_path = "data/processed.h5"

feature_cols = {'date': 'datetime64', 'serial_number': str,
                'smart_197_raw': 'float', 'smart_9_raw': 'float', 'smart_241_raw': 'float', 'smart_187_raw': 'float',
                'failure': 'bool'}

smart_col_names = [col_name for col_name in feature_cols.keys() if col_name.startswith('smart')]

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
    group_copy = group.copy()
    if group_copy.failure.any():
        group_copy['days_to_failure'] = (group_copy.date.max() - group_copy.date).dt.days
    else:
        group_copy['days_to_failure'] = -1
    return group_copy

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


def main():
    with pd.HDFStore(hdf5_path) as hdf:
        hash_bucket_count = 20
        with tqdm(total=len(os.listdir(raw_data_path))) as progress_bar:
            for filename in os.listdir(raw_data_path):
                progress_bar.update()
                progress_bar.set_description(f"Parsing raw file {progress_bar.n}/{progress_bar.total}: {filename}")
                filepath = os.path.join(raw_data_path, filename)
                df = pd.read_csv(filepath, parse_dates=['date'])
                df = df[feature_cols.keys()].astype(feature_cols)
                hash_col = df.serial_number.apply(hash) % hash_bucket_count
                groupby = df.groupby(hash_col)
                for group_id, group_df in groupby:
                    hdf.append(f"processed/group{group_id}", group_df)

        dataset_names = next(hdf.walk('/processed'))[2]
        with tqdm(total=len(dataset_names)) as process_bar:
            for dataset_name in dataset_names:
                df = hdf.get(f"processed/{dataset_name}")
                means = df[smart_col_names].mean(axis=0)
                stddevs = df[smart_col_names].std(axis=0)
                df[smart_col_names] = (df[smart_col_names] - means) / stddevs
                groupby = df.groupby('serial_number')
                df = groupby.apply(add_days_to_failure_column_to_group)
                hdf.put(f"/normalized/{dataset_name}", df)
                process_bar.update()
                process_bar.set_description(f"Normalizing hdf dataset {process_bar.n}/{process_bar.total}: {dataset_name}")










if __name__ == "__main__":
    main()

    print(time_measuring_data)