import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from utils import Lemmatizer, tokenize_subtokenize_comment, tokenize_clean_code


lem = Lemmatizer()


class Example:
    def __init__(self, id, old_comment, new_comment, old_code, new_code):
        self.id = id
        self.old_comment = old_comment
        self.new_comment = new_comment
        self.old_code = old_code
        self.new_code = new_code


def my_diff(old_tokens, new_tokens):
    group_num = 0
    old = []
    new = []
    for group in SequenceMatcher(None, old_tokens, new_tokens).get_grouped_opcodes(0):
        group_num += 1
        for tag, i1, i2, j1, j2 in group:
            if tag in {'replace', 'delete'}:
                for token in old_tokens[i1:i2]:
                    old.append(token)
            if tag in {'replace', 'insert'}:
                for token in new_tokens[j1:j2]:
                    new.append(token)
    return group_num, old, new


def is_example_valid(example: Example):
    """ return False means the example is turely modified 
    """
    old_tokens, old_subtokens = tokenize_subtokenize_comment(example.old_comment)
    new_tokens, new_subtokens = tokenize_subtokenize_comment(example.new_comment)
    token_group_num, old_diff_tokens, new_diff_tokens = my_diff(old_tokens, new_tokens)
    subtoken_group_num, old_diff_subtokens, new_diff_subtokens = my_diff(old_subtokens, new_subtokens)
    token_diff = f"{token_group_num}: {str(old_diff_tokens)} -> {str(new_diff_tokens)}"
    subtoken_diff = f"{subtoken_group_num}: {str(old_diff_subtokens)} -> {str(new_diff_subtokens)}"
    if subtoken_group_num == 0:
        return True, token_diff, subtoken_diff
    if (subtoken_group_num == 1 and
            len(old_diff_tokens) <= 1 and len(new_diff_tokens) <= 1 and
            len(old_diff_subtokens) <= 1 and len(new_diff_subtokens) <= 1):
        flag = True
        for token in old_diff_tokens + new_diff_tokens:
            if token.lower() not in ['a', 'an', 'the', 'in', 'on', 'at']:
                flag = False
                break
        if flag:
            return True, token_diff, subtoken_diff
        if len(old_diff_subtokens) == 1 and len(new_diff_subtokens) == 1:
            if (check_same_semantic(old_diff_subtokens[0], new_diff_subtokens[0]) or
                    check_typo(old_diff_subtokens[0], new_diff_subtokens[0])):
                tokens = tokenize_clean_code(example.old_code)
                if old_diff_tokens[0] not in tokens:
                    return True, token_diff, subtoken_diff
    return False, token_diff, subtoken_diff


def check_same_semantic(token1, token2):
    return lem.lemmatize(token1) == lem.lemmatize(token2)


def check_typo(token1, token2):
    if token1.isalpha() and token2.isalpha():
        edit_dist = 0
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, token1, token2).get_opcodes():
            if tag in ['replace', 'delete', 'insert']:
                edit_dist += max(i2 - i1, j2 - j1)
        return edit_dist <= 2
    return False


def positive_data_clean(path):
    examples = []
    positive_sample_id = []
    # for file in ['train', 'valid', 'test']:
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            positive_sample_id.append(obj['sample_id'])
            _id = f"{obj['idx']}_{obj['sample_id']}"
            examples.append(Example(_id, obj['src_desc'], obj['dst_desc'], obj['src_method'], obj['dst_method']))
    with open('positive_sample_id.json', 'w') as f:  # used to select reliable samples
        json.dump(positive_sample_id, f)
    positive_id = []
    for example in examples:
        flag, tokens_diff, subtoken_diff = is_example_valid(example)
        if not flag:
            positive_id.append(example.id)
            continue
    with open('positive_id.json', 'w') as f:
        json.dump(positive_id, f)
    print("Successfully Executed Label Correction Script!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str, help='path to original updater data, this file will make "positive_id.json" and "positive_sample_id.json" files in current dir!')
    args = parser.parse_args()
    path = Path(args.path)
    positive_data_clean(path)
