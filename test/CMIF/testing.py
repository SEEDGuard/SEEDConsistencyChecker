import os
import sys

# Add the parent directory of "core" to the Python path
sys.path.append(os.getcwd())
from core.CMIF.cmiFinder import CmiFinder


def main():
    input_dir = "data/input"
    output_dir = "data/output"

    checker: CmiFinder = CmiFinder()
    checker.consistency_checker(data_dir=input_dir, dest_dir=output_dir)


if __name__ == "__main__":
    main()
