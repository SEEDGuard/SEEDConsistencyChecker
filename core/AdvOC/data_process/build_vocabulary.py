
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import pickle

import numpy as np
from dpu_utils.mlutils import Vocabulary
# import fasttext

from diff_utils import get_edit_keywords

START = '<sos>'
END = '<eos>'
VOCAB_CUTOFF_PCT = 5
LENGTH_CUTOFF_PCT = 95
MAX_VOCAB_SIZE = 50000


def build_vocabulary(train_data, output_file, embed_file):
    nl_lengths = []
    code_lengths = []
    ast_lengths = []

    token_counter = Counter()

    for ex in train_data:
        old_nl_sequence = ex.old_comment_subtokens
        token_counter.update(set(old_nl_sequence))
        nl_lengths.append(len(old_nl_sequence))

        code_sequence = ex.span_diff_code_subtokens
        token_counter.update(set(code_sequence))
        code_lengths.append(len(code_sequence))

    # max_nl_length = int(np.percentile(np.asarray(sorted(nl_lengths)), LENGTH_CUTOFF_PCT))
    max_nl_length = 160
    max_code_length = 380
    max_vocab_extension = 0
    max_ast_length = 380

    nl_counts = np.asarray(sorted(token_counter.values()))
    nl_threshold = int(np.percentile(nl_counts, VOCAB_CUTOFF_PCT)) + 1

    edit_keywords = get_edit_keywords()
    nl_vocab = Vocabulary.create_vocabulary(
        tokens=edit_keywords, max_size=MAX_VOCAB_SIZE, count_threshold=1, add_pad=True)
    type_to_id = json.load(open('./ast_type_dict.json'))
    type_to_id = Counter(**type_to_id)
    type_to_id.update(['SuperRoot'])
    nl_vocab.update(type_to_id, MAX_VOCAB_SIZE, count_threshold=1)
    nl_vocab.update(token_counter, MAX_VOCAB_SIZE, nl_threshold)

    obj = dict(
        max_nl_length=max_nl_length,
        max_code_length=max_code_length,
        max_vocab_extension=max_vocab_extension,
        max_ast_length=max_ast_length,
        nl_vocab=nl_vocab.__dict__,
    )
    with open(output_file, 'w') as f:
        json.dump(obj, f, indent=2)
    # id_to_token = nl_vocab.id_to_token
    # ft_model = fasttext.load_model('/mnt/data1/kingxu/fastText/cc.en.300.bin')
    # embedding_list = []
    # for s in id_to_token:
    #     embedding_list.append(ft_model.get_word_vector(s))
    # embed = np.vstack(embedding_list)
    # print(embed.shape)
    # embed[0] = 0.0 # for pad
    # pickle.dump(embed, open(embed_file, 'wb'))


class Example:
    def __init__(self, old_comment_subtokens, span_diff_code_subtokens) -> None:
        self.old_comment_subtokens = old_comment_subtokens
        self.span_diff_code_subtokens = span_diff_code_subtokens


def read_examples(path):
    examples = []
    with open(path / 'train_seq.jsonl') as f:
        for l in f:
            obj = json.loads(l)
            examples.append(Example(obj['ocs'], obj['sdcs']))
    return examples


def main(path):
    print('Starting building vocabulary: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
    train_examples = read_examples(path)
    build_vocabulary(train_examples, path / 'vocab.json', path / 'embed.pkl')
    print('Terminating building vocabulary: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str, help='path to training file dir, this file will generate a "vocab.json" in this path!')
    args = parser.parse_args()
    path = Path(args.path)
    main(path)
