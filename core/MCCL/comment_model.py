import subprocess

from core.MCCL.constants import *
from  . import *


class Comment_Model:
    
    data_path = DEFAULT_DATA_PATH
    def __init__(self, input_path = None):
        if (input_path != None):
            self.data_path = input_path
        pass
    def evaluate_test_dataset(self, stats, output_path=None, num_batches=None):
        # Predict the inconsistencies
        args = ["--attend_code_sequence_states", "--features", 
                "--model_path=detect_attend_code_sequence_states_features_cl.pkl.gz", 
                "--model_name=detect_attend_code_sequence_states_features_cl", "--test_mode",
                "--data_path=" + self.data_path]
        subprocess.run(["python3", "core/MCCL/run_comment_model.py"] + args)