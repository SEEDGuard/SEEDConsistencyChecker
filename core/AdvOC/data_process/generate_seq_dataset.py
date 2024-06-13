import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

import jsonlines

from diff_utils import (DELETE, DELETE_END, INSERT, INSERT_END, KEEP, KEEP_END,
                        REPLACE_END, REPLACE_NEW, REPLACE_OLD,
                        compute_code_diffs, is_edit_keyword)
from external_cache import is_java_keyword, is_operator
from high_level_feature_extractor import get_change_labels, get_method_elements
from utils import subtokenize_token, tokenize_clean_code, tokenize_comment


def tokenize_subtokenize_comment(comment_raw):
    """
    input:comment_raw
    output:comment_subtokens, nl_subtoken_labels, nl_subtoken_indices, nl_token_map, nl_subtoken_map
    """
    tokens = tokenize_comment(comment_raw)  # List[str]
    labels, indices = [], []
    token_map, subtoken_map = [], []
    all_subtokens = []
    for token in tokens:
        subtokens = subtokenize_token(token)
        all_subtokens.extend(subtokens)
        token_map.append(subtokens)  # 保存的是每个token对应的subtokens
        if len(subtokens) == 1:
            labels.append(0)
            indices.append(0)
            subtoken_map.append([token])  # 保存的是每个subtoken对应的token(属于哪个token)
        else:
            for s, subtoken in enumerate(subtokens):
                labels.append(1)  # 0和1分别对应是单subtoken成词还是多subtoken成词
                indices.append(s)  # 对应的是subtoken在token中的位置
                subtoken_map.append([token])
    return all_subtokens, labels, indices, token_map, subtoken_map


def tokenize_subtokenize_diff_code(old_code_raw, new_code_raw):
    """
    input:old_code_raw, new_code_raw
    output:old_code_subtokens, new_code_subtokens, span_diff_code_subtokens, token_diff_code_subtokens, diff_labels, diff_indices, diff_token_map, diff_subtoken_map
    """

    def tokenize_subtokenize_code(code_raw):
        """
        输入:code_raw
        输出:code_subtokens, code_subtoken_labels, code_subtoken_indices, code_token_map, code_subtoken_map
        """
        code_tokens = tokenize_clean_code(code_raw)  # List[str]
        labels, indices = [], []
        token_map, subtoken_map = [], []
        all_subtokens = []
        for token in code_tokens:
            subtokens = subtokenize_token(token)
            all_subtokens.extend(subtokens)
            token_map.append(subtokens)
            if len(subtokens) == 1:
                labels.append(0)
                indices.append(0)
                subtoken_map.append([token])
            else:
                for s, subtoken in enumerate(subtokens):
                    labels.append(1)
                    indices.append(s)
                    subtoken_map.append([token])
        return code_tokens, all_subtokens, labels, indices, token_map, subtoken_map

    (
        old_code_tokens,
        old_code_subtokens,
        old_code_subtoken_labels,
        old_code_subtoken_indices,
        old_code_token_map,
        old_code_subtoken_map,
    ) = tokenize_subtokenize_code(old_code_raw)
    (
        new_code_tokens,
        new_code_subtokens,
        new_code_subtoken_labels,
        new_code_subtoken_indices,
        new_code_token_map,
        new_code_subtoken_map,
    ) = tokenize_subtokenize_code(new_code_raw)

    span_diff_code_tokens, _, _ = compute_code_diffs(old_code_tokens, new_code_tokens)
    span_diff_code_subtokens, token_diff_code_subtokens, _ = compute_code_diffs(old_code_subtokens, new_code_subtokens)

    diff_labels, diff_indices = [], []
    diff_token_map, diff_subtoken_map = [], []

    for token in span_diff_code_tokens:
        if is_edit_keyword(token):
            diff_token_map.append([token])
        else:
            subtokens = subtokenize_token(token)
            diff_token_map.append(subtokens)

    for edit_type, o_start, o_end, n_start, n_end in SequenceMatcher(None, old_code_subtokens, new_code_subtokens).get_opcodes():
        if edit_type == 'equal':
            diff_labels.extend([0] + old_code_subtoken_labels[o_start:o_end] + [0])
            diff_indices.extend([0] + old_code_subtoken_indices[o_start:o_end] + [0])
            diff_subtoken_map.append([KEEP])
            diff_subtoken_map.extend(old_code_subtoken_map[o_start:o_end])
            diff_subtoken_map.append([KEEP_END])
        elif edit_type == 'replace':
            diff_labels.extend([0] + old_code_subtoken_labels[o_start:o_end] + [0] + new_code_subtoken_labels[n_start:n_end] + [0])
            diff_indices.extend([0] + old_code_subtoken_indices[o_start:o_end] + [0] + new_code_subtoken_indices[n_start:n_end] + [0])
            diff_subtoken_map.append([REPLACE_OLD])
            diff_subtoken_map.extend(old_code_subtoken_map[o_start:o_end])
            diff_subtoken_map.append([REPLACE_NEW])
            diff_subtoken_map.extend(new_code_subtoken_map[n_start:n_end])
            diff_subtoken_map.append([REPLACE_END])
        elif edit_type == 'insert':
            diff_labels.extend([0] + new_code_subtoken_labels[n_start:n_end] + [0])
            diff_indices.extend([0] + new_code_subtoken_indices[n_start:n_end] + [0])
            diff_subtoken_map.append([INSERT])
            diff_subtoken_map.extend(new_code_subtoken_map[n_start:n_end])
            diff_subtoken_map.append([INSERT_END])
        else:
            diff_labels.extend([0] + old_code_subtoken_labels[o_start:o_end] + [0])
            diff_indices.extend([0] + old_code_subtoken_indices[o_start:o_end] + [0])
            diff_subtoken_map.append([DELETE])
            diff_subtoken_map.extend(old_code_subtoken_map[o_start:o_end])
            diff_subtoken_map.append([DELETE_END])
    return (
        old_code_subtokens,
        new_code_subtokens,
        span_diff_code_subtokens,
        token_diff_code_subtokens,
        diff_labels,
        diff_indices,
        diff_token_map,
        diff_subtoken_map,
    )


def generate_clean_dataset(old_data_path, new_data_path):
    # splits = ['train', 'full_valid', 'valid', 'test']
    try:
        with open('id_delete.json') as f:
            deleted_id = set(json.load(f))
    except:
        print('no id_delete.json')
        deleted_id = []
    try:
        with open('positive_id.json') as f:
            positive_id = set(json.load(f))
    except:
        print('no positive_id.json')
        positive_id = []
    try:
        with open('positive_sample_id.json') as f:
            positive_sample_id = set(json.load(f))
    except:
        print('no positive_sample_id.json')
        positive_sample_id = []
        
    old_file_name = Path(old_data_path).stem

    # for s in splits:
    rf1 = jsonlines.open(new_data_path / f'{old_file_name}_seq.jsonl', 'w')
    with open(old_data_path) as f:
        for line in f:
            if not line.strip():
                break
            obj = json.loads(line)
            # metadata
            id = f"{obj['idx']}_{obj['sample_id']}"
            if id in deleted_id:
                continue
            sample_id = int(obj['sample_id'])
            label = int(id in positive_id)
            old_label = int(obj['label'])
            confidence = int(sample_id in positive_sample_id)

            # process
            old_comment_raw = obj['src_desc']
            (
                old_comment_subtokens,
                old_nl_subtoken_labels,
                old_nl_subtoken_indices,
                old_nl_token_map,
                old_nl_subtoken_map,
            ) = tokenize_subtokenize_comment(old_comment_raw)

            old_code_raw = obj['src_method']
            new_code_raw = obj['dst_method']
            (
                old_code_subtokens,
                new_code_subtokens,
                span_diff_code_subtokens,
                token_diff_code_subtokens,
                edit_span_subtoken_labels,
                edit_span_subtoken_indices,
                edit_span_token_map,
                edit_span_subtoken_map
            ) = tokenize_subtokenize_diff_code(old_code_raw, new_code_raw)

            old_comment_subtoken_labels = old_nl_subtoken_labels
            old_comment_subtoken_indices = old_nl_subtoken_indices
            old_comment_token_map = [token[0] for token in old_nl_subtoken_map]
            start_end = [0, max(1, len(old_comment_subtokens))]

            old_method_elements = get_method_elements(old_code_raw.split('\n'))
            new_method_elements = get_method_elements(new_code_raw.split('\n'))
            code_change_labels = get_change_labels(token_diff_code_subtokens)

            old_return_type_tokens = list(set(old_method_elements['subtoken']['return_type']))
            new_return_type_tokens = list(set(new_method_elements['subtoken']['return_type']))
            old_return_line_tokens = list(set([t for t in old_method_elements['subtoken']['return_statement'] if not is_java_keyword(t) and not is_operator(t)]))
            new_return_line_tokens = list(set([t for t in new_method_elements['subtoken']['return_statement'] if not is_java_keyword(t) and not is_operator(t)]))
            insert_code_tokens = list(set(code_change_labels['<INSERT>']))
            keep_code_tokens = list(set(code_change_labels['<KEEP>']))
            delete_code_tokens = list(set(code_change_labels['<DELETE>']))
            replace_old_code_tokens = list(set(code_change_labels['<REPLACE_OLD>']))
            replace_new_code_tokens = list(set(code_change_labels['<REPLACE_NEW>']))
            rf1.write(dict(
                id=id,
                label=label,
                ol=old_label,
                cf=confidence,
                ocs=old_comment_subtokens,
                ocsl=old_comment_subtoken_labels,
                ocsi=old_comment_subtoken_indices,
                octm=old_comment_token_map,
                s_e=start_end,
                sdcs=span_diff_code_subtokens,
                essl=edit_span_subtoken_labels,
                essi=edit_span_subtoken_indices,
                ortt=old_return_type_tokens,
                nrtt=new_return_type_tokens,
                orlt=old_return_line_tokens,
                nrlt=new_return_line_tokens,
                ict=insert_code_tokens,
                kct=keep_code_tokens,
                dct=delete_code_tokens,
                roct=replace_old_code_tokens,
                rnct=replace_new_code_tokens
            ))
    print("Successfully Generated Seq Dataset!")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_path', required=True, type=str, help='path to original dataset')
    parser.add_argument('--new_path', required=True, type=str, help='path to the cleaned dataset')
    args = parser.parse_args()
    old_data_path = Path(args.old_path)
    new_data_path = Path(args.new_path)
    generate_clean_dataset(old_data_path, new_data_path)
