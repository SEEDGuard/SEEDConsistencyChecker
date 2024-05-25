import argparse
import sys
from datetime import datetime


import torch
sys.path.append('comment_update')
from detection_module import DetectionModule
from data_loader import get_data_splits
from constants import *

def build_model(model_path, manager):
    """ Builds the appropriate model, with task-specific modules."""
    model = DetectionModule(model_path, manager)
    
    return model

def load_model(model_path):
    """Loads a pretrained model from model_path."""
    print('Loading model from: {}'.format(model_path))
    sys.stdout.flush()
    if torch.cuda.is_available():
        model = torch.load(model_path)
        model.torch_device_name = 'gpu'
        model.cuda()
        for c in model.children():
            c.cuda()
    else:
        model = torch.load(model_path, map_location='cpu')
        model.torch_device_name = 'cpu'
        model.cpu()
        for c in model.children():
            c.cpu()
    return model

def evaluate(model, test_examples, model_name):
    """Runs evaluation over a given model."""
    print('Evaluating {} examples'.format(len(test_examples)))
    sys.stdout.flush()
    predictions = model.run_evaluation(test_examples, model_name)
    
    with open("predictions_output.txt", "w") as file:
        for idx, result in enumerate(predictions):
            file.write("Example {}: {}\n".format(idx, result))
    print("Output printed to predictions_output.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attend_code_sequence_states', action='store_true', help='attend to sequence-based code hidden states for detection')
    parser.add_argument('--attend_code_graph_states', action='store_true', help='attend to graph-based code hidden states for detection')
    parser.add_argument('--features', action='store_true', help='concatenate lexical and linguistic feats to code/comment input embeddings')
    parser.add_argument('--positive_only', action='store_true', help='whether to train on only inconsistent examples')
    parser.add_argument('--test_mode', action='store_true', help='whether to run evaluation')
    parser.add_argument('--model_path', help='path to save model(training) or path to save model(evaluation)')
    parser.add_argument('--model_name', help='name of model (used to save model output)')
    parser.add_argument('--data_path', help='path to input dataset')
    args = parser.parse_args()

    # We should only need all examples to be test examples right?
    # Also all of these datapoints have labels for if they are inconsistent or not
    test_examples, high_level_details, _ = get_data_splits(args.data_path)
    
    print('Test: {}'.format(len(test_examples)))
    res = test_examples[:1]
 
    # print result
    print("The first element in the list is: " + str(res))

    if not args.attend_code_sequence_states and not args.attend_code_graph_states:
        raise ValueError('Please specify attention states for detection')

    if args.test_mode:
        print('Starting evaluation: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
        
        model = load_model(args.model_path)
        evaluate(model, test_examples, args.model_name)
        
        print('Terminating evaluation: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
