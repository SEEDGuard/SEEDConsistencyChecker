# Import your method from core and pass user data
import argparse
import sys
sys.path.append('../..')
from core.deep_justintime.deep_justintime_model import *
# add the imports for your methods

def get_method(method_name):
    # We need to validate here if the input method_name exist in our method or not
    # check your method name
    if method_name.lower() == 'deep_justintime_eval_test_ds':
        # return your method class 
        return Deep_JustInTime_Model()
    
    # Add more checkers as needed
    else:
        raise ValueError(f"Invalid method name: {method_name}")

def main():
    parser = argparse.ArgumentParser(description='Check for inconsistencies in provided test dataset with specified methods.')
    parser.add_argument('--input_dir', type=str,
                        help='Path to the input dataset', required=False)
    parser.add_argument('--output_dir', type=str,
                        help='Path to the output directory', required=False)
    parser.add_argument('--method', type=str,
                        help='Name of the method to use')
    parser.add_argument('--num_batches', type=int, help='Analyzes the first n batches (based on BATCH_SIZE in utils/constants.py) within the test dataset', required=False)
    parser.add_argument('--stats', action='store_true', help='Adds stats about the data being tested (accuracy, precision, etc.)')
    args = parser.parse_args()

    checker = get_method(args.method)
    checker.evaluate_test_dataset(args.stats, args.output_dir, num_batches=args.num_batches)
    
    #call the method object and pass user data

if __name__ == "__main__":
    main()



