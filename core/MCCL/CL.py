import argparse
import os
import sys
sys.path.append('comment_update')

import json
import numpy as np
from datetime import datetime
import torch
from detection_module import DetectionModule
from data_loader import get_data_splits
from module_manager import ModuleManager
from constants import *
import torch.nn.functional
from cleanlab.pruning import get_noise_indices
from cleanlab.latent_estimation import (compute_confident_joint, estimate_latent, estimate_joint)



def load_model(model_path):
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
    sys.stdout.flush()
    model.run_evaluation(test_examples, model_name, True)


def clean_data():
    train_examples, valid_examples, test_examples, high_level_details, comment_type_str = get_data_splits()
    model = load_model(args.model_path)
    if args.dataset == 'train':
        evaluate(model, train_examples, args.model_name)
    else:
        evaluate(model, valid_examples, args.model_name)

    open_file_prob = os.path.join(DETECTION_DIR, 'probs.json')
    with open(open_file_prob, "r") as f:
        probs = json.load(f)

    open_file_label = os.path.join(DETECTION_DIR, 'gold_labels.json')
    with open(open_file_label, "r") as f:
        gold_labels = json.load(f)

    open_file_data = os.path.join(DATA_PATH, comment_type_str, args.dataset + '.json')
    with open(open_file_data, "r") as f:
        data = json.load(f)

    noisy_labels = np.array(gold_labels)
    predicted_probabilities = np.array(probs)

    if args.threshold == 'None':
        th = None
    else:
        th = float(args.threshold)

    confident_joint, indices = compute_confident_joint(
        s=noisy_labels,
        psx=predicted_probabilities,
        thresholds=th,
        return_indices_of_off_diagonals=True
    )

    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=noisy_labels,
        py_method='cnt',
        converge_latent_estimates=False,
    )

    ordered_label_errors = get_noise_indices(
        s=noisy_labels,
        psx=predicted_probabilities,
        inverse_noise_matrix=inv_noise_matrix,
        confident_joint=confident_joint,
        prune_method='prune_by_noise_rate',
        sorted_index_method='normalized_margin',
        frac_noise=1
    )

    if args.method == 'C':
        ordered_label_errors = indices

    data_out = []
    label_out = []
    for e, ex in enumerate(data):
        if e not in ordered_label_errors:
            data_out.append(data[e])
            label_out.append(gold_labels[e])

    write_file = os.path.join(DATA_PATH, comment_type_str, args.dataset + '_clean_{}.json'.format(th))
    with open(write_file, "w+") as f:
        json.dump(data_out, f)

    print("Raw Nosiy data: {}".format(len(data)))
    print("Clean data: {}".format(len(data_out)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help='C or Q', default='Q')
    parser.add_argument('--dataset', help='train or valid')
    parser.add_argument('--threshold', help='percentage to prune, from 0.1 to 0.9', default='None')
    parser.add_argument('--model_path', help='path to save model (training) or path to saved model (evaluation)')
    parser.add_argument('--model_name', help='name of model (used to save model output)')
    parser.add_argument('--attend_code_sequence_states', action='store_true', help='attend to sequence-based code hidden states for detection')
    parser.add_argument('--attend_code_graph_states', action='store_true', help='attend to graph-based code hidden states for detection')
    parser.add_argument('--features', action='store_true', help='concatenate lexical and linguistic feats to code/comment input embeddings')
    args = parser.parse_args()
    clean_data()
