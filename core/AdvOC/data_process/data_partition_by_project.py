import json
from pathlib import Path

from tqdm import tqdm


def read_all_data(dir):
    """ (train,full_valid,test)
    """
    dir = Path(dir)
    split = ['c_train', 'u_train', 'full_valid', 'test']
    seq_lines = []
    for s in split:
        with open(dir / f'{s}_seq.jsonl') as f:
            for l in f:
                seq_lines.append(l)
    tree_lines = []
    for s in split:
        with open(dir / f'{s}_tree.jsonl') as f:
            for l in f:
                tree_lines.append(l)
    assert len(seq_lines) == len(tree_lines)
    print(len(tree_lines), flush=True)
    return seq_lines, tree_lines


def split_by_project(seq_lines, tree_lines, dir, id_by_project_file):
    with open(id_by_project_file) as f:
        cleaned_ids = json.load(f)
    test_ids = set(cleaned_ids['test'])
    valid_ids = set(cleaned_ids['valid'])
    train1, full_valid1, test1 = [], [], []
    train2, full_valid2, test2 = [], [], []
    for seql, treel in tqdm(zip(seq_lines, tree_lines)):
        cid = seql.split(' ', 2)[1].strip('",')
        if cid in test_ids:
            test1.append(seql.rstrip())
            test2.append(treel.rstrip())
        elif cid in valid_ids:
            full_valid1.append(seql.rstrip())
            full_valid2.append(treel.rstrip())
        else:
            train1.append(seql.rstrip())
            train2.append(treel.rstrip())

    print(f'train: {len(train1)}, full_valid: {len(full_valid1)}, test: {len(test1)}')
    del seq_lines, tree_lines
    dir = Path(dir)
    valid_id_lines = {}
    for l1, l2 in zip(full_valid1, full_valid2):
        cid = l1.split(' ', 2)[1].strip('",')
        valid_id_lines[cid] = (l1, l2)
    test_id_lines = {}
    for l1, l2 in zip(test1, test2):
        cid = l1.split(' ', 2)[1].strip('",')
        test_id_lines[cid] = (l1, l2)
    full_valid1, full_valid2 = [], []
    for i in cleaned_ids['valid']:
        full_valid1.append(valid_id_lines[i][0])
        full_valid2.append(valid_id_lines[i][1])
    test1, test2 = [], []
    for i in cleaned_ids['test']:
        test1.append(test_id_lines[i][0])
        test2.append(test_id_lines[i][1])
    with open(dir / 'train_seq.jsonl', 'w') as f:
        for l in train1:
            f.write(l + '\n')
    with open(dir / 'full_valid_seq.jsonl', 'w') as f:
        for l in full_valid1:
            f.write(l + '\n')
    with open(dir / 'test_seq.jsonl', 'w') as f:
        for l in test1:
            f.write(l + '\n')
    with open(dir / 'train_tree.jsonl', 'w') as f:
        for l in train2:
            f.write(l + '\n')
    with open(dir / 'full_valid_tree.jsonl', 'w') as f:
        for l in full_valid2:
            f.write(l + '\n')
    with open(dir / 'test_tree.jsonl', 'w') as f:
        for l in test2:
            f.write(l + '\n')


if __name__ == '__main__':
    dir = './' # TODO: the cleaned dataset
    new_dir = './'  # TODO: the dir to save split_by_project cleaned dataset
    id_by_project_file = './'  # TODO: this file can be find in the cleaned dataset
    lines1, lines2 = read_all_data(dir)
    split_by_project(lines1, lines2, new_dir, id_by_project_file)
