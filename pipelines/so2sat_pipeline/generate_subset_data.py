import h5py
import argparse


def main(input_file, output_file, num_samples):
    # Open the original HDF5 file for reading
    with h5py.File(input_file, 'r') as f:
        label_len = len(f['label'])

        if num_samples > label_len:
            num_samples = label_len

        input_val = f['sen1'][:num_samples]
        label_val = f['label'][:num_samples]

    # Create a new HDF5 file for writing
    with h5py.File(output_file, 'w') as tiny_f:
        # Create datasets in the new file and copy data
        tiny_f.create_dataset('sen1', data=input_val)
        tiny_f.create_dataset('label', data=label_val)

    print(f'Data saved to {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a specified number of samples from an HDF5 file.')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input HDF5 file')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the output HDF5 file')
    parser.add_argument('--num-samples', type=int, default=320, help='Number of samples to extract (default: 320)')

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.num_samples)
