import pandas as pd
import argparse
import os
import torch
import re

parser = argparse.ArgumentParser()
parser.add_argument('-s', dest='rows_in_memory', type=int, default=None,
                    help="maximum amount of rows that is kept in memory at any one time")
args = parser.parse_args()

raw_data_path = "../../data/raw"
features_path = "../../data/features"

def feature_extract(df):
    return df[['smart_5_normalized', 'smart_187_normalized', 'smart_188_normalized', 'smart_197_normalized',
        'smart_198_normalized']]

def save_as_pytorch(df, path):
    torch.save(torch.tensor(feature_extract(df).values), path)

def parse_file(input_path, df_to_feature_file):
    base_filename = re.findall(r"/([^/]*)\.csv", input_path)[0]
    feature_file_path = os.path.join(features_path, base_filename)
    if args.rows_in_memory is None:
        df = pd.read_csv(input_path)
        df_to_feature_file(df, feature_file_path)
    else:
        skiprows = 1
        still_rows_left = True
        output_file_suffix = 0
        while still_rows_left:
            df = pd.read_csv(input_path, skiprows=range(1, skiprows), nrows=args.rows_in_memory)
            skiprows += args.rows_in_memory
            if len(df) < args.rows_in_memory:
                still_rows_left = False
            feature_file_path = os.path.join(features_path, f"{base_filename}_{output_file_suffix}")
            df_to_feature_file(df, feature_file_path)
            output_file_suffix += 1

def main():
    os.mkdir(features_path)
    for filename in os.listdir(raw_data_path):
        parse_file(os.path.join(raw_data_path, filename), save_as_pytorch)


if __name__ == "__main__":
    main()