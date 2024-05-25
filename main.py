# Import your method from core and pass user data

import argparse

from core.CMIF.cmiFinder import CmiFinder
from core.MCCL.comment_model import *

# add the imports for your methods

def get_method(method_name, input_path):
    # We need to validate here if the input method_name exist in our method or not
    # check your method name
    # if method_name.lower() == 'method_name':
    #     return your method class

    if method_name.lower() == 'cmif':
        # return your method class
        return CmiFinder()
    if method_name.lower() == 'mccl':
        return Comment_Model(input_path)

    # Add more checkers as needed
    else:
        raise ValueError(f"Invalid method name: {method_name}")


def main():
    parser = argparse.ArgumentParser(description='Check for inconsistencies in your dataset with specified methods.')
    parser.add_argument('--input_dir', type=str,
                        help='Path to the input dataset')
    parser.add_argument('--output_dir', type=str,
                        help='Path to the output directory')
    parser.add_argument('--method', type=str,
                        help='Name of the method to use')
    parser.add_argument('--num_batches', type=int, help='Analyzes the first n batches (based on BATCH_SIZE in utils/constants.py) within the test dataset', required=False)
    parser.add_argument('--stats', action='store_true', help='Adds stats about the data being tested (accuracy, precision, etc.)')
    args = parser.parse_args()

    checker = get_method(args.method)

    # call the method object and pass user data
    checker.consistency_checker(data_dir=args.input_dir, dest_dir=args.output_dir)


if __name__ == "__main__":
    main()
