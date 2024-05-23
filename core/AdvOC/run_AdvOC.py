"""
Adversarial obsolete comment detection
"""

import argparse
import os
from pathlib import Path
import random
import sys
from datetime import datetime

import numpy as np
import torch

from module_manager import ModuleManager
from predictive_adversary_networks import PredictiveAdversaryNetworks


def setup_seed(seed):
    """
    Setup seed for reproducibility.
    ref: https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"


def load_model(model, model_path):
    """Loads a pretrained model from model_path."""
    print('Loading model from: {}'.format(model_path))
    sys.stdout.flush()
    model.load_model(model_path)
    if torch.cuda.is_available():
        model.discriminator.cuda()
        model.classifier.cuda()
    else:
        model.discriminator.cpu()
        model.classifier.cpu()
    return model


def evaluate(model, model_path,input_dir,output_dir):
    model.run_evaluation(model_path, input_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', default=True, action='store_true', help='use edit seq for detection')
    parser.add_argument('--tree',default=True,  action='store_true', help='use edit tree for detection')
    parser.add_argument('--script', default=True, action='store_true', help='use edit script for detection')
    parser.add_argument('--test_mode',default=True, action='store_true', help='whether to run evaluation')
    parser.add_argument('--model_path', default = 'model', help='path to save model (training) or path to saved model (evaluation)')
    parser.add_argument('--model_file', default='core/AdvOC/model/classifier.pt', help='the model to be evaluated!')
    parser.add_argument('--input_dir', help='path to the input directory')
    parser.add_argument('--output_dir', help='path to the output directory')
    
    args = parser.parse_args()

    setup_seed(22)

    if args.test_mode:
        print('Starting evaluation: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
        print(type(args.seq), args.seq)
        print(type(args.tree), args.tree)
        print(type(args.script), args.script)
        
        manager1 = ModuleManager(args.seq, args.tree, args.script)
        manager2 = ModuleManager(args.seq, args.tree, args.script)
        manager1.initialize()
        manager2.initialize()
        model = PredictiveAdversaryNetworks(args.model_path, manager1, manager2)
        # if args.model_file:
        model.load_classifier(args.model_file)
        # else:
        #     model = load_model(model, args.model_path)
        evaluate(model, args.model_path, args.input_dir, args.output_dir)
        print('Terminating evaluation: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))

