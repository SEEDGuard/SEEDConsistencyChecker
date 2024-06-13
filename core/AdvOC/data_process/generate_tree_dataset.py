

import argparse
import json
from pathlib import Path
from subprocess import PIPE, Popen

import jsonlines


class Node():
    def __init__(self) -> None:
        self.type = None
        self.label = ''
        self.pos = None
        self.children = []
        self.father = None
        self.idx = None

    def __str__(self) -> str:
        return f'{self.idx} {self.type} {self.pos} {self.father.idx if self.father else -1} {self.label}'

    def __repr__(self) -> str:
        return self.__str__()

    def unique_id(self) -> str:
        return '{} {}'.format(self.type, self.pos)

    def print_tree(self, depth):
        s = '  ' * depth + str(self) + '\n'
        for child in self.children:
            s += child.print_tree(depth + 1)
        return s

    def dumps_with_type_id(self, ast_type_dict):
        return f'{self.idx} {ast_type_dict.get(self.type, 1)} {self.pos} {self.father.idx if self.father else -1} {self.label}'


def gen_ast(ast_str):
    try:
        ast = json.loads(ast_str)
    except:
        return []
    nodes = process_ast(ast['root'])
    cu = nodes[0]
    cu.idx = 0
    nodes_new = [cu]
    cur_node = cu
    idx = 1
    for node in nodes[1:]:
        if node == '^':
            cur_node = cur_node.father
        else:
            node.idx = idx
            node.father = cur_node
            cur_node.children.append(node)
            cur_node = node
            nodes_new.append(node)
            idx += 1
    return nodes_new


def process_ast(ast):
    nodes = []
    # if ast['type'] == 'Javadoc':
    #     return nodes
    node = Node()
    node.type = ast['type']
    node.pos = ast['pos']
    if 'label' in ast:
        node.label = ast['label']
    elif node.type == 'NullLiteral':
        node.label = 'null'
    elif node.type == 'ThisExpression':
        node.label = 'this'
    nodes.append(node)
    for child in ast['children']:
        ns = process_ast(child)
        if ns:
            nodes += ns
            nodes.append('^')
    return nodes


def get_node_idx_map(nodes):
    node_idx = {}
    for node in nodes:
        node_idx[node.unique_id()] = node.idx
    return node_idx


def process_act(act_str, node_idx1, node_idx2):
    """
    node_idx1是old tree
    node_idx2是new tree
    return: (matches, inserts, deletes, updates, moves)
    """
    act = json.loads(act_str)
    matches = dict()
    for m in act['matches']:
        if ':' not in m['src']:
            matches[node_idx1.get(m['src'], -1)] = node_idx2.get(m['dest'], -1)
        else:
            src = f"{m['src'].split(':', 1)[0]} {m['src'].rsplit(' ', 1)[-1]}"
            dest = f"{m['dest'].split(':', 1)[0]} {m['dest'].rsplit(' ', 1)[-1]}"
            matches[node_idx1.get(src, -1)] = node_idx2.get(dest, -1)
    inserts = set()
    deletes = set()
    updates = dict()
    moves = dict()
    for a in act['actions']:
        if a['action'].startswith('insert'):
            if ':' not in a['tree']:
                node = a['tree']
            else:
                node = f"{a['tree'].split(':', 1)[0]} {a['tree'].rsplit(' ', 1)[-1]}"
            inserts.add(node_idx2.get(node, -1))
        elif a['action'].startswith('delete'):
            if ':' not in a['tree']:
                node = a['tree']
            else:
                node = f"{a['tree'].split(':', 1)[0]} {a['tree'].rsplit(' ', 1)[-1]}"
            deletes.add(node_idx1.get(node, -1))
        elif a['action'].startswith('update'):
            if ':' not in a['tree']:
                node = a['tree']
                dest = a['dest']
            else:
                node = f"{a['tree'].split(':', 1)[0]} {a['tree'].rsplit(' ', 1)[-1]}"
                dest = f"{a['dest'].split(':', 1)[0]} {a['dest'].rsplit(' ', 1)[-1]}"
            updates[node_idx1.get(node, -1)] = node_idx2.get(dest, -1)
        elif a['action'].startswith('move'):
            if ':' not in a['tree']:
                node = a['tree']
                dest = a['dest']
            else:
                node = f"{a['tree'].split(':', 1)[0]} {a['tree'].rsplit(' ', 1)[-1]}"
                dest = f"{a['dest'].split(':', 1)[0]} {a['dest'].rsplit(' ', 1)[-1]}"
            moves[node_idx1.get(node, -1)] = node_idx2.get(dest, -1)
    return matches, inserts, deletes, updates, moves


def dumps_nodes(nodes, ast_type_dict):
    res = []
    for node in nodes:
        res.append(node.dumps_with_type_id(ast_type_dict))
    return res


def simple_tree_diff(obj, ast_type_dict):
    """
    return: nodes1, nodes2, matches, inserts, deletes, updates, moves
    """
    nodes1 = gen_ast(obj['old_tree'])
    nodes2 = gen_ast(obj['new_tree'])
    # node_idx_map1 = get_node_idx_map2(nodes1, ast_type_dict)
    # node_idx_map2 = get_node_idx_map2(nodes2, ast_type_dict)
    node_idx_map1 = get_node_idx_map(nodes1)
    node_idx_map2 = get_node_idx_map(nodes2)
    act = process_act(obj['act'], node_idx_map1, node_idx_map2)
    return (
        dumps_nodes(nodes1, ast_type_dict),
        dumps_nodes(nodes2, ast_type_dict),
        [[k, v] for k, v in act[0].items()],
        list(act[1]),
        list(act[2]),
        [[k, v] for k, v in act[3].items()],
        [[k, v] for k, v in act[4].items()],
    )


def generate_clean_tree_dataset(old_data_path, new_data_path):
    # splits = ['train', 'full_valid', 'valid', 'test']
    try:
        with open('id_delete.json') as f:
            deleted_id = set(json.load(f))
    except:
        print('no id_delete.json')
        deleted_id = []
    
    old_file_name = Path(old_data_path).stem
    # for s in splits:
    writer = jsonlines.open(new_data_path / 'code.jsonl', 'w')
    with open(old_data_path) as f:
        for line in f:
            if not line.strip():
                break
            obj = json.loads(line)
            cid = f'{obj["idx"]}_{obj["sample_id"]}'
            if cid in deleted_id:
                continue
            writer.write(dict(pr='', cid=f'{obj["idx"]}_{obj["sample_id"]}', old_code=obj['src_method'], new_code=obj['dst_method']))
    Popen(['java', '-jar', './gumtree_demo.jar', '-i', str(new_data_path / 'code.jsonl'), '-o', str(new_data_path / 'tmp.jsonl')],
            stdout=PIPE, stderr=PIPE).communicate()
    Popen(['rm', '-rf', str(new_data_path / 'code.jsonl')], stdout=PIPE, stderr=PIPE).communicate()
    ast_type_dict = json.load(open('./ast_type_dict.json'))
    with open(new_data_path / 'tmp.jsonl') as f, jsonlines.open(new_data_path / f'{old_file_name}_tree.jsonl', 'w') as writer:
        for l in f:
            obj = json.loads(l)
            nodes1, nodes2, match, insert, delete, update, move = simple_tree_diff(obj, ast_type_dict)
            writer.write(dict(cid=obj['cid'], no1=nodes1, no2=nodes2, mat=match, ins=insert, dele=delete, upd=update, mov=move))
    Popen(['rm', '-rf', str(new_data_path / 'tmp.jsonl')], stdout=PIPE, stderr=PIPE).communicate()
    print("Successfully Generated Tree Dataset!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_path', required=True, type=str, help='path to original dataset')
    parser.add_argument('--new_path', required=True, type=str, help='path to the cleaned dataset')
    args = parser.parse_args()
    old_data_path = Path(args.old_path)
    new_data_path = Path(args.new_path)
    generate_clean_tree_dataset(old_data_path, new_data_path)
