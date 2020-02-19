import argparse
from utils_file import read_json, write_json, check_make_dir
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    args = parser.parse_args()
    input_data = read_json(args.input_file)
    file_name = args.input_file.split('/')[-1]
    small_data = input_data[:15]
    check_make_dir(args.output_dir)
    output_file = os.path.join(args.output_dir, file_name)
    write_json(small_data, output_file)