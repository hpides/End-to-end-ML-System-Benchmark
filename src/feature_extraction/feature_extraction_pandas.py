import pandas as pd
import argparse
import os
import torch
import re

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', dest='rows_in_memory', type=int, default=None,
                    help="maximum amount of rows that is kept in memory at any time, None if no such limitation")
parser.add_argument('-t', '--target', dest='target', default='pytorch', choices=['pytorch', 'numpy', 'tensorflow'],
                    help='format of saved feature tensors')
args = parser.parse_args()

raw_data_path = "../../data/raw"
features_path = "../../data/features"

def feature_extract(df):
    return df[['smart_5_normalized', 'smart_187_normalized', 'smart_188_normalized', 'smart_197_normalized',
        'smart_198_normalized']]

def save_features_to_file(df, write_path):
    if args.target == 'pytorch':
        torch.save(torch.tensor(feature_extract(df).values), write_path)

def parse_and_save_files():
    os.mkdir(features_path)
    for read_filename in os.listdir(raw_data_path):
        read_filepath = os.path.join(raw_data_path, read_filename)
        read_file_base_name = re.findall(r"(.*)\.csv", read_filename)[0]

        if args.rows_in_memory is None:
            write_filepath = os.path.join(features_path, read_file_base_name)
            df = pd.read_csv(read_filepath)
            save_features_to_file(df, write_filepath)
        else:
            skiprows = 1
            rows_left_to_read = True
            output_file_suffix = 0
            while rows_left_to_read:
                df = pd.read_csv(read_filepath, skiprows=range(1, skiprows), nrows=args.rows_in_memory)
                skiprows += args.rows_in_memory
                if len(df) < args.rows_in_memory:
                    rows_left_to_read = False
                write_filename = f"{read_file_base_name}_{output_file_suffix}"
                write_filepath = os.path.join(features_path, write_filename)
                save_features_to_file(df, write_filepath)
                output_file_suffix += 1

if __name__ == "__main__":
    parse_and_save_files()