import argparse
from collections import defaultdict
from difflib import unified_diff
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str, help='path to original data, this file will make a "id_delete.json" file in current dir!')
    args = parser.parse_args()

    # splits = [ 'valid', 'test']

    count = 0
    code_diff_old_comment_dict = defaultdict(list)
    dir = Path(args.path)
    # for s in splits:
    with open(dir) as f:
        for l in f:
            count += 1
            obj = json.loads(l)
            old_code_lines = obj['src_method'].split('\n')
            new_code_lines = obj['dst_method'].split('\n')
            old_code_lines = [l.strip() for l in old_code_lines if l.strip() != '']
            new_code_lines = [l.strip() for l in new_code_lines if l.strip() != '']
            diff = '\n'.join(list(unified_diff(old_code_lines, new_code_lines, lineterm='', n=100))[3:])
            code_diff_old_comment_dict[diff + '\n' + obj['src_desc'].replace('\n', ' ')].append(f"{obj['idx']}_{obj['sample_id']}: {int(obj['label'])}")

    print("Total Instances:",count)

    id_keep = []
    id_all = []
    iid = set()
    for v in code_diff_old_comment_dict.values():
        kid = ''
        klable = -1
        for s in v:
            id, label = s.split(': ')
            id_all.append(str(id))
            lable = int(label)
            if lable > klable:
                kid = id
                klable = lable
            if id.split('_')[1] in iid:
                kid = id
                klable = 2
        id_keep.append(str(kid))
        iid.add(kid.split('_')[1])
    id_delete = list(set(id_all) - set(id_keep))
    json.dump(id_delete, open('id_delete.json', 'w'))
