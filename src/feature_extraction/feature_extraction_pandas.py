import h5py
from tqdm import tqdm
import pandas as pd
import os
import numpy as np

raw_data_path = "data/raw"
pandas_h5_path = "data/pandas.h5"
h5py_h5_path = "data/h5py.h5"

feature_col_types = {'date': 'datetime64', 'serial_number': str,
                     'smart_197_raw': 'float', 'smart_9_raw': 'float', 'smart_241_raw': 'float', 'smart_187_raw': 'float',
                     'failure': 'bool'}
feature_col_names = list(feature_col_types.keys())
smart_col_names = [name for name in feature_col_names if name.startswith('smart')]
smart_col_count = len(smart_col_names)

def add_days_to_failure_col_to_group(group):
    group_copy = group.copy()
    if group_copy.failure.any():
        group_copy['days_to_failure'] = (group_copy.date.max() - group_copy.date).dt.days
    else:
        group_copy['days_to_failure'] = -1
    return group_copy

def main():
    entry_count = 0
    pandas_hdf = pd.HDFStore(pandas_h5_path)
    h5py_hdf = h5py.File(h5py_h5_path, 'a')

    # read raw csv files and filter out unnecessary columns
    hash_bucket_count = 20
    with tqdm(total=len(os.listdir(raw_data_path))) as progress_bar:
        for filename in os.listdir(raw_data_path):
            progress_bar.update()
            progress_bar.set_description(f"Parsing raw file {progress_bar.n}/{progress_bar.total}: {filename}")
            filepath = os.path.join(raw_data_path, filename)
            df = pd.read_csv(filepath, parse_dates=['date'])
            entry_count += len(df)
            df = df[feature_col_names].astype(feature_col_types)
            hash_col = df.serial_number.apply(hash) % hash_bucket_count
            groupby = df.groupby(hash_col)
            for group_id, group_df in groupby:
                pandas_hdf.append(f"/group{group_id}", group_df)

    h5_features = h5py_hdf.create_dataset('/features', shape=(entry_count, smart_col_count), dtype='float')
    h5_labels_numeric = h5py_hdf.create_dataset('/labels_numeric', shape=(entry_count), dtype='int')
    offset = 0

    # normalize, impute missing values, add numeric days_to_failure column
    dataset_names = next(pandas_hdf.walk('/'))[2]
    with tqdm(total=len(dataset_names)) as process_bar:
        for dataset_name in dataset_names:
            df = pandas_hdf.get(f"/{dataset_name}")
            groupby = df.groupby('serial_number')
            df = groupby.apply(add_days_to_failure_col_to_group)
            features = df[smart_col_names].values
            means = np.nanmean(features, axis=0)
            idx = np.where(np.isnan(features))
            features[idx] = np.take(means, idx[1])
            stddevs = np.std(features, axis=0)
            features = (features - means) / stddevs
            labels_numeric = df.days_to_failure.values
            h5_features[offset:offset+len(features), :] = features
            h5_labels_numeric[offset:offset + len(features)] = labels_numeric
            offset += len(features)
            process_bar.update()
            process_bar.set_description(f"Normalizing hdf dataset {process_bar.n}/{process_bar.total}: {dataset_name}")

    # categorize labels (days_to_failure column)
    labels_numeric_dataset = h5py_hdf['labels_numeric']
    labels_series = pd.Series(labels_numeric_dataset)
    fail = labels_series[labels_series != -1]
    cut, bins = pd.qcut(fail, 3, retbins=True)
    cut = pd.cut(labels_series, bins)
    no_failure_interval = pd.Interval(-1, -1, closed='both')
    cut.cat.add_categories(no_failure_interval, inplace=True)
    cut.fillna(no_failure_interval, inplace=True)
    one_hot_labels = pd.get_dummies(cut).astype('bool').values
    h5py_hdf.create_dataset('/labels_one_hot', data=one_hot_labels)

    pandas_hdf.close()
    h5py_hdf.close()


if __name__ == "__main__":
    main()