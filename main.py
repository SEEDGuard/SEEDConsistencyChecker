# Import your method from core and pass user data

import argparse

from test.CMIF.testing import predict_code_t5


# add the imports for your methods

def get_method(method_name):
    # We need to validate here if the input method_name exist in our method or not
    # check your method name
    # if method_name.lower() == 'method_name':
    #     return your method class

    if method_name.lower() == 'cmif':
        # return your method class
        return predict_code_t5
    
    # Add more checkers as needed
    else:
        raise ValueError(f"Invalid method name: {method_name}")

def main():
    parser = argparse.ArgumentParser(description='Check Inconsistency in your dataset with a specified methods.')
    parser.add_argument('--input_dir', type=str,
                        help='Path to the input dataset')
    parser.add_argument('--output_dir', type=str,
                        help='Path to the output directory')
    parser.add_argument('--method', type=str,
                        help='Name of the method to use')

    args = parser.parse_args()

    checker = get_method(args.method)
    
    #call the method object and pass user data
    checker(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()



