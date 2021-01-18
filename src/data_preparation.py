import re

import h5py
from tqdm import tqdm
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter

raw_data_path = "data/raw"
pandas_h5_path = "data/pandas.h5"
h5py_h5_path = "data/h5py.h5"

feature_col_types = {'date': 'datetime64', 'serial_number': str,
                     'smart_197_raw': 'float', 'smart_9_raw': 'float', 'smart_241_raw': 'float', 'smart_187_raw': 'float',
                     'failure': 'bool'}
feature_col_names = list(feature_col_types.keys())
smart_col_names = [name for name in feature_col_names if name.startswith('smart')]
smart_col_count = len(smart_col_names)

pandas_hdf = pd.HDFStore(pandas_h5_path)
h5py_hdf = h5py.File(h5py_h5_path, 'a')

def parse_raw_csv_files():
    hash_bucket_count = 20
    with tqdm(total=len(os.listdir(raw_data_path))) as progress_bar:
        for filename in os.listdir(raw_data_path):
            progress_bar.update()
            progress_bar.set_description(f"Parsing raw file {progress_bar.n}/{progress_bar.total}: {filename}")
            filepath = os.path.join(raw_data_path, filename)
            df = pd.read_csv(filepath, parse_dates=['date'])
            df = df[feature_col_names].astype(feature_col_types)
            hash_col = df.serial_number.apply(hash) % hash_bucket_count
            groupby = df.groupby(hash_col)
            for group_id, group_df in groupby:
                pandas_hdf.append(f"/group{group_id}", group_df, min_itemsize={'serial_number': 21})

def add_days_to_failure_col_to_group(group):
    group_copy = group.copy()
    if group_copy.failure.any():
        group_copy['days_to_failure'] = (group_copy.date.max() - group_copy.date).dt.days
    else:
        group_copy['days_to_failure'] = -1
    return group_copy

def transfer_from_pandas_to_h5py():
    dataset_lengths = [int(length) for length in re.findall("nrows->(\d*)", pandas_hdf.info())]
    entry_count = sum(dataset_lengths)
    print(entry_count)

    X_h5 = h5py_hdf.create_dataset('/X', shape=(entry_count, smart_col_count), dtype='float')
    y_h5 = h5py_hdf.create_dataset('/y', shape=(entry_count), dtype='int')
    offset = 0
    with tqdm(total=len(pandas_hdf.keys())) as process_bar:
        for dataset_name in pandas_hdf.keys():
            df = pandas_hdf.get(dataset_name)
            groupby = df.groupby('serial_number')
            df = groupby.apply(add_days_to_failure_col_to_group)
            X_chunk = df[smart_col_names].values
            y_chunk = df['days_to_failure'].values
            X_h5[offset:offset+len(X_chunk), :] = X_chunk
            y_h5[offset:offset+len(y_chunk)] = y_chunk
            offset += len(X_chunk)
            process_bar.update()
            process_bar.set_description(
                f"Transferring dataset {process_bar.n}/{process_bar.total} to h5py : {dataset_name}")

def normalization_and_categorization():
    X_h5 = h5py_hdf['X']
    y_h5 = h5py_hdf['y']

    X = X_h5[:, :]
    y_numeric = y_h5[:]

    # normalization
    means = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(means, idx[1])
    stddevs = np.std(X, axis=0)
    X = (X - means) / stddevs

    # categorize labels
    y_numeric = pd.Series(y_numeric)
    fail = y_numeric[y_numeric != -1]
    bin_count = 3
    bin_labels = np.arange(bin_count)
    cut, bins = pd.qcut(fail, bin_count, retbins=True, labels=bin_labels)
    print(f"Bins used for categorization of labels: {bins} with labels: {bin_labels}")
    y_categoric = pd.cut(y_numeric, bins, labels=bin_labels)
    y_categoric.cat.add_categories(-1, inplace=True)
    y_categoric.fillna(-1, inplace=True)
    y = y_categoric

    X_h5[:, :] = X
    y_h5[:] = y

def dataset_splitting_and_resampling():
    X = h5py_hdf['X'][:,:]
    y = h5py_hdf['y'][:]

    # split dataset
    print('Splitting into train and test set')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    h5py_hdf.create_dataset('X_test', data=X_test)
    h5py_hdf.create_dataset('y_test', data=y_test)

    print('Resampling')
    # undersampling
    print(f"Original distribution of labels: {dict(Counter(y_train))}")

    undersampling_strat = {-1: int(6e6)}
    undersampler = RandomUnderSampler(sampling_strategy=undersampling_strat)
    print(f"Undersampling non-failures using {undersampler}")
    X_train, y_train = undersampler.fit_resample(X_train, y_train)
    print(f"Distribution of y after undersampling: {dict(Counter(y_train))}")

    # SMOTE
    oversampling_strat = {2: int(2e6), 1: int(2e6), 0: int(2e6)}
    oversampler = SMOTE(sampling_strategy=oversampling_strat)
    print(f"Oversampling failures using {oversampler}")
    X_train, y_train = oversampler.fit_resample(X_train, y_train)
    print(f"Distribution of y after oversampling: {dict(Counter(y_train))}")

    h5py_hdf.create_dataset('X_train', data=X_train)
    h5py_hdf.create_dataset('y_train', data=y_train)

def main():
    parse_raw_csv_files()
    transfer_from_pandas_to_h5py()
    normalization_and_categorization()
    dataset_splitting_and_resampling()
    pandas_hdf.close()
    h5py_hdf.close()

if __name__ == "__main__":
    main()