# 一些内联标签 {@link } {@code } {@linkplain } {@value } {@codesnippet } 其他的不用关注
# 一些测试用例:
# Generates a code sample for using {@link FileAsyncClient#startCopyWithResponse(String, Map)}
# modified time to set to {@link LayerConfiguration#DEFAULT_MODIFIED_TIME}.
# Returns the color of the {@code R.attr.colorControlHighlight} attribute in the overridden style.
# Returns a {@linkplain ClassDesc} given a descriptor string for a class, interface, array, or primitive type.
# Validates and returns the underlying {@link LogicalType} of the given {@link Schema.Field}.
# In order to set the matrix to a lookat transformation without post-multiplying it, use {@link #setLookAt(Vector3dc, Vector3dc, Vector3dc)}.
# {@codesnippet com.azure.storage.file.fileAsyncClient.upload#bytebuf-long-int-filerangewritetype}


import enum
import json
import re
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple

import nltk
import numpy as np
import torch
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from numpy import ndarray
from tqdm import tqdm

from constants import MAX_SCRIPT_LENGTH, MAX_SUBTOKENS
from diff_utils import (DELETE, INSERT, KEEP, REPLACE_NEW, REPLACE_OLD,
                        is_edit_keyword)
from embedding_store import EmbeddingStore
from external_cache import (get_num_code_features, get_num_nl_features,
                            is_java_keyword, is_operator, stop_words, tags)

nltk.data.path.insert(0, '/data/share/kingxu/nltk_data')


def subtokenize_token(token):
    """
    return: List[str] str is lowercase
    """
    if is_edit_keyword(token):
        return [token]
    # 处理下划线
    token = token.replace('_', ' ')
    # 处理CamelCase
    token = re.sub(r'([A-Z][a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', token))
    # 处理数字
    token = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', token)
    token = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', token)

    try:
        curr = [c for c in re.findall(r"[a-zA-Z0-9_]+|[^\sa-zA-Z0-9_]", token.encode('ascii', errors='ignore').decode()) if len(c) > 0]
    except:
        curr = token.split()
    subtokens = [c.lower() for c in curr]

    return subtokens


@enum.unique
class SrcType(Enum):
    KEEP = 0
    INSERT = 1
    DELETE = 2
    REPLACE_OLD = 3
    REPLACE_NEW = 4
    MOVE = 5


class Lemmatizer:
    def __init__(self):
        self.cache = {}
        self.tag_dict = {'J': wordnet.ADJ,
                         'N': wordnet.NOUN,
                         'V': wordnet.VERB,
                         'R': wordnet.ADV}
        self.wnl = WordNetLemmatizer()

    def lemmatize(self, word):
        if word in self.cache:
            return self.cache[word]
        else:
            lemma = self.wnl.lemmatize(word, self.get_wordnet_pos(word))
            self.cache[word] = lemma
            return lemma

    def get_wordnet_pos(self, word):
        tag = pos_tag([word])[0][1][0].upper()
        return self.tag_dict.get(tag, wordnet.NOUN)


# lem = Lemmatizer()


class EditScript(NamedTuple):
    action: str
    old_value: List[str]
    new_value: List[str]
    old_path: List[str]
    new_path: List[str]


def build_script(action, old_node, new_node):
    old_value = []
    new_value = []
    old_path = []
    new_path = []
    if old_node is not None:
        old_value.append(old_node.attribute)
        if old_node.is_leaf:
            old_value.extend(subtokenize_token(old_node.value))
        node = old_node.parent
        while node is not None:
            old_path.append(node.attribute)
            node = node.parent
    if new_node is not None:
        new_value.append(new_node.attribute)
        if new_node.is_leaf:
            new_value.extend(subtokenize_token(new_node.value))
        node = new_node.parent
        while node is not None:
            new_path.append(node.attribute)
            node = node.parent
    return EditScript(action, old_value, new_value, old_path, new_path)


class DiffTreeNode:
    def __init__(self, value, attribute, src, is_leaf):
        self.value = value
        self.node_id = -1
        self.parents = []  # 可能有多个父节点
        self.attribute = attribute
        self.src = src
        self.is_leaf = is_leaf
        self.children = []
        self.prev_siblings = []  # 因为两棵树合并, 可能有多个前驱节点
        self.next_siblings = []  # 因为两棵树合并, 可能有多个后继节点
        self.aligned_neighbors = []
        self.action_type = None
        self.prev_tokens = []
        self.next_tokens = []
        self.subtokens = []

        self.subtoken_children = []
        self.subtoken_parents = []
        self.prev_subtokens = []
        self.next_subtokens = []

    def to_json(self):
        return {
            'value': self.value,
            'node_id': self.node_id,
            'parent_ids': [p.node_id for p in self.parents],
            'attribute': self.attribute,
            'src': self.src,
            'is_leaf': self.is_leaf,
            'children_ids': [c.node_id for c in self.children],
            'prev_sibling_ids': [p.node_id for p in self.prev_siblings],
            'next_sibling_ids': [n.node_id for n in self.next_siblings],
            'aligned_neighbor_ids': [n.node_id for n in self.aligned_neighbors],
            'action_type': self.action_type,
        }

    @property
    def is_identifier(self):
        return self.is_leaf and self.attribute == 'SimpleName'


class DiffAST:
    def __init__(self, ast_root, scripts=[]):
        self.node_cache = set()
        self.root = ast_root
        self.nodes = []
        self.traverse(self.root)
        self.scripts = scripts

    def traverse(self, curr_node):  # 可能有环
        if curr_node not in self.node_cache:
            self.node_cache.add(curr_node)
            curr_node.node_id = len(self.nodes)
            self.nodes.append(curr_node)
            for child in curr_node.subtoken_children:
                self.traverse(child)
            for child in curr_node.children:
                self.traverse(child)

    def to_json(self):
        return [n.to_json() for n in self.nodes]

    @property
    def leaves(self):
        return [n for n in self.nodes if n.is_leaf]

    @classmethod
    def from_json(cls, obj, id_to_type):
        """
        首先根据obj生成两个AST, 然后根据以old AST构建建DiffAST的关系,
        再遍历new AST, 将new AST中的关系添加到DiffAST中,关键就是相同节点的复用.
        """
        old_nodes = load_nodes(obj['no1'], id_to_type, src='old')
        old_ast = AST(old_nodes[0])
        new_nodes = load_nodes(obj['no2'], id_to_type, src='new')
        new_ast = AST(new_nodes[0])

        scripts = []

        matches = {}
        old_actions = {}
        new_actions = {}
        old_len = len(old_nodes)
        new_len = len(new_nodes)
        for k, v in obj['mat']:
            k -= 5
            v -= 5
            if 0 <= k < old_len and 0 <= v < new_len:
                matches[v] = k
        for k, v in obj['upd']:
            k -= 5
            v -= 5
            if 0 <= k < old_len and 0 <= v < new_len:
                matches[v] = k
                old_actions[k] = 'Update'
                scripts.append(build_script('Update', old_nodes[k], new_nodes[v]))
        for k, v in obj['mov']:
            k -= 5
            v -= 5
            if 0 <= k < old_len and 0 <= v < new_len:
                matches[v] = k
                old_actions[k] = 'Move'
                scripts.append(build_script('Move', old_nodes[k], new_nodes[v]))
        for k in obj['dele']:
            k -= 5
            if 0 <= k < old_len:
                old_actions[k] = 'Delete'
                scripts.append(build_script('Delete', old_nodes[k], None))
        for k in obj['ins']:
            k -= 5
            if 0 <= k < new_len:
                new_actions[k] = 'Insert'
                scripts.append(build_script('Insert', None, new_nodes[k]))

        old_diff_nodes = []
        for n in old_nodes:
            old_diff_node = DiffTreeNode(n.value, n.attribute, n.src, n.is_leaf)
            if n.node_id in old_actions:  # 应该是一个存idx: action的字典
                # Insert, Delete, Move, Update
                old_diff_node.action_type = old_actions[n.node_id]
            old_diff_nodes.append(old_diff_node)

        for n, old_node in enumerate(old_nodes):
            # assert old_node.node_id == n
            old_diff_node = old_diff_nodes[n]
            if old_node.parent:
                old_diff_node.parents.append(old_diff_nodes[old_node.parent.node_id])

            for c in old_node.children:
                old_diff_node.children.append(old_diff_nodes[c.node_id])

            if old_node.prev_sibling:
                old_diff_node.prev_siblings.append(old_diff_nodes[old_node.prev_sibling.node_id])

            if old_node.next_sibling:
                old_diff_node.next_siblings.append(old_diff_nodes[old_node.next_sibling.node_id])

        new_diff_nodes = []
        for n, new_node in enumerate(new_nodes):
            if n in matches:  # matches应该是用new_idx查找old_idx,包括match,move,update
                old_diff_node = old_diff_nodes[matches[n]]
                if new_node.value == old_diff_node.value:
                    new_diff_node = old_diff_node
                    new_diff_node.src = 'both'
                    new_diff_nodes.append(new_diff_node)
                else:
                    new_diff_node = DiffTreeNode(new_node.value, new_node.attribute, new_node.src, new_node.is_leaf)
                    new_diff_node.aligned_neighbors.append(old_diff_node)
                    old_diff_node.aligned_neighbors.append(new_diff_node)
                    new_diff_node.action_type = old_diff_node.action_type
                    if n in new_actions:
                        new_diff_node.action_type = new_actions[n]
                    new_diff_nodes.append(new_diff_node)
            else:
                new_diff_node = DiffTreeNode(new_node.value, new_node.attribute, new_node.src, new_node.is_leaf)
                if n in new_actions:
                    new_diff_node.action_type = new_actions[n]
                new_diff_nodes.append(new_diff_node)

        for n, new_node in enumerate(new_nodes):
            # assert new_node.node_id == n
            new_diff_node = new_diff_nodes[n]
            if new_node.parent and new_diff_nodes[new_node.parent.node_id] not in new_diff_node.parents:
                new_diff_node.parents.append(new_diff_nodes[new_node.parent.node_id])
            for c in new_node.children:
                if new_diff_nodes[c.node_id] not in new_diff_node.children:
                    new_diff_node.children.append(new_diff_nodes[c.node_id])
            if new_node.prev_sibling and new_diff_nodes[new_node.prev_sibling.node_id] not in new_diff_node.prev_siblings:
                new_diff_node.prev_siblings.append(new_diff_nodes[new_node.prev_sibling.node_id])
            if new_node.next_sibling and new_diff_nodes[new_node.next_sibling.node_id] not in new_diff_node.next_siblings:
                new_diff_node.next_siblings.append(new_diff_nodes[new_node.next_sibling.node_id])
        super_root = DiffTreeNode('SuperRoot', 'SuperRoot', 'both', False)
        super_root.children.append(old_diff_nodes[0])
        old_diff_nodes[0].parents.append(super_root)
        if old_diff_nodes[0] != new_diff_nodes[0]:
            super_root.children.append(new_diff_nodes[0])
            new_diff_nodes[0].parents.append(super_root)
        diff_ast = cls(super_root)
        for node in diff_ast.nodes:
            if len(node.children) == 0:
                # 特殊处理QualifiedName,他算不上原子节点,将它变为非叶节点
                if node.attribute == 'QualifiedName':
                    node.is_leaf = False
                    names = node.value.split('.')
                    for name in names:
                        new_node = DiffTreeNode(name, 'SimpleName', node.src, True)
                        new_node.subtokens = subtokenize_token(name)
                        new_node.action_type = node.action_type
                        new_node.parents.append(node)
                        if len(node.children) > 0:
                            node.children[-1].next_siblings.append(new_node)
                            new_node.prev_siblings.append(node.children[-1])
                        node.children.append(new_node)
                else:
                    node.is_leaf = True

                    # curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', node.value).split()
                    # new_curr = []
                    # for c in curr:
                    #     by_symbol = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip())
                    #     new_curr = new_curr + by_symbol
                    # node.subtokens = [s.lower() for s in new_curr]

                    # 更好的分词方法
                    node.subtokens = subtokenize_token(node.value)

                    # 不为子词添加节点试试
                    # if len(node.subtokens) > 1:
                    #     for s in node.subtokens:
                    #         sub_node = DiffTreeNode(s, '', node.src, True)
                    #         sub_node.action_type = node.action_type
                    #         sub_node.subtoken_parents.append(node)
                    #         if len(node.subtoken_children) > 0:
                    #             node.subtoken_children[-1].next_subtokens.append(sub_node)
                    #             sub_node.prev_subtokens.append(node.subtoken_children[-1])
                    #         node.subtoken_children.append(sub_node)
                    # node.value = node.value.lower() # 在这里先不转换,因为后面要判断是否和注释中Token完全一致
        return cls(super_root, scripts)


class GraphMethodBatch:
    def __init__(self, graph_ids, value_lookup_ids, src_type_ids, root_ids, is_internal,
                 edges, num_graphs, num_nodes, node_features, node_positions, num_nodes_per_graph):
        self.graph_ids = graph_ids
        self.value_lookup_ids = value_lookup_ids
        self.src_type_ids = src_type_ids
        self.root_ids = root_ids
        self.is_internal = is_internal
        self.edges = edges
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.node_positions = node_positions
        self.num_nodes_per_graph = num_nodes_per_graph


def initialize_graph_method_batch(num_edges):
    return GraphMethodBatch(
        graph_ids=[],
        value_lookup_ids=[],
        src_type_ids=[],
        root_ids=[],
        is_internal=[],
        edges=[[] for _ in range(num_edges)],
        num_graphs=0,
        num_nodes=0,
        node_features=[],
        node_positions=[],
        num_nodes_per_graph=[]
    )


def load_nodes(res, i_type, src):
    nodes = []
    for line in res[5:]:
        idx, type, pos, father, label = line.split(' ', 4)
        type = i_type[int(type)]
        is_leaf = False
        value = type
        if label:
            value = label
            is_leaf = True
        alignment_id = None
        location_id = pos  # 这里是一处不同
        idx = int(idx) - 5
        father_idx = int(father) - 5
        if father_idx < 0:
            father_node = None
        else:
            father_node = nodes[father_idx]
        node = XMLNode(value, idx, father_node, type, alignment_id, location_id, src, is_leaf)

        if father_node is not None:
            father_node.children.append(node)
        nodes.append(node)
    return nodes


class XMLNode:
    def __init__(self, value, node_id, parent, attribute,
                 alignment_id, location_id, src, is_leaf=True):
        self.value = value
        self.node_id = node_id
        self.parent = parent
        self.attribute = attribute
        self.alignment_id = alignment_id
        self.location_id = location_id
        self.src = src
        self.is_leaf = is_leaf
        self.children = []
        self.pseudo_children = []
        self.prev_sibling = None
        self.next_sibling = None

    def print_node(self):
        parent_value = None
        if self.parent:
            parent_value = self.parent.value

        print('{}: {} ({}, {})'.format(self.node_id, self.value, parent_value, len(self.children)))
        for c in self.children:
            c.print_node()


class AST:
    def __init__(self, ast_root):
        self.root = ast_root
        # self.nodes = []
        self.traverse(ast_root)

    def traverse(self, curr_node):
        # self.nodes.append(curr_node)
        for c, child_node in enumerate(curr_node.children):
            if c > 0:
                child_node.prev_sibling = curr_node.children[c-1]
            if c < len(curr_node.children) - 1:
                child_node.next_sibling = curr_node.children[c+1]
            self.traverse(child_node)

    # @property
    # def leaves(self):
        # return [n for n in self.nodes if n.is_leaf]


class Example(NamedTuple):
    id: str
    label: int
    old_context_subtokens: List[str]
    nl_start: int
    nl_end: int
    span_diff_code_subtokens: List[str]
    nl_feature: ndarray
    code_feature: ndarray
    diff_ast: DiffAST  # or None
    node_features: ndarray  # or None
    token_match: ndarray  # or None


def open_files(file_names):
    return [open(f, 'r') for f in file_names]


def build_nl_features(old_nl_sequence, max_nl_length, obj, detail_tmp, tokenization):
    frequency_map = dict()
    for tok in old_nl_sequence:
        if tok not in frequency_map:
            frequency_map[tok] = 0
        frequency_map[tok] += 1

    pos_tags = pos_tag(old_nl_sequence)
    pos_tag_indices = []
    for _, t in pos_tags:
        if t in tags:
            pos_tag_indices.append(tags.index(t))
        else:
            pos_tag_indices.append(tags.index('OTHER'))

    (
        old_return_line_terms, new_return_line_terms, return_line_intersection,
        old_set, new_set, intersection,
        insert_code_tokens, keep_code_tokens, delete_code_tokens,
        replace_old_code_tokens, replace_new_code_tokens
    ) = detail_tmp
    nl_subtoken_labels = obj['ocsl']
    nl_subtoken_indices = obj['ocsi']

    features = np.zeros((max_nl_length, get_num_nl_features()), dtype=np.int32)
    for i in range(len(old_nl_sequence)):
        if i >= max_nl_length:
            break
        token = old_nl_sequence[i]  # 这里不需要lower,UNK为大写,lower后导致key error
        if token in intersection:
            features[i][0] = True
        elif token in old_set:
            features[i][1] = True
        elif token in new_set:
            features[i][2] = True
        else:
            features[i][3] = True

        if token in return_line_intersection:
            features[i][4] = True
        elif token in old_return_line_terms:
            features[i][5] = True
        elif token in new_return_line_terms:
            features[i][6] = True
        else:
            features[i][7] = True

        features[i][8] = token in insert_code_tokens
        features[i][9] = token in keep_code_tokens
        features[i][10] = token in delete_code_tokens
        features[i][11] = token in replace_old_code_tokens
        features[i][12] = token in replace_new_code_tokens
        features[i][13] = token in stop_words
        features[i][14] = frequency_map[token] > 1

        features[i][15] = nl_subtoken_labels[i]
        features[i][16] = nl_subtoken_indices[i]
        features[i][17 + pos_tag_indices[i]] = 1

    return features.astype(np.float32)


def build_code_features(code_sequence, max_code_length, obj, detail_tmp, tokenization):
    (
        old_return_line_terms, new_return_line_terms, return_line_intersection,
        old_set, new_set, intersection,
        _, _, _, _, _
    ) = detail_tmp

    features = np.zeros((max_code_length, get_num_code_features()), dtype=np.int32)

    old_nl_tokens = set(obj['ocs'])  # TODO: 这里是context了,要不要只包含old_nl
    last_command = None

    subtoken_labels = tokenization['edit_span_subtoken_labels']
    subtoken_indices = tokenization['edit_span_subtoken_indices']

    for i, token in enumerate(code_sequence):
        if i >= max_code_length:
            break
        if token in intersection:
            features[i][0] = True
        elif token in old_set:
            features[i][1] = True
        elif token in new_set:
            features[i][2] = True
        else:
            features[i][3] = True

        if token in return_line_intersection:
            features[i][4] = True
        elif token in old_return_line_terms:
            features[i][5] = True
        elif token in new_return_line_terms:
            features[i][6] = True
        else:
            features[i][7] = True

        if is_edit_keyword(token):
            features[i][8] = True
        if is_java_keyword(token):
            features[i][9] = True
        if is_operator(token):
            features[i][10] = True
        if token in old_nl_tokens:
            features[i][11] = True

        if not is_edit_keyword(token):
            if last_command == KEEP:
                features[i][12] = 1
            elif last_command == INSERT:
                features[i][13] = 1
            elif last_command == DELETE:
                features[i][14] = 1
            elif last_command == REPLACE_NEW:
                features[i][15] = 1
            else:
                features[i][16] = 1
        else:
            last_command = token

        features[i][17] = subtoken_labels[i]
        features[i][18] = subtoken_indices[i]

    return features.astype(np.float32)


def get_num_node_features():
    return 4


def build_tmp_example(obj1, obj4, id_to_type):
    old_comment_subtokens = obj1['old_comment_subtokens']
    span_diff_code_subtokens = obj1['span_diff_code_subtokens']
    # diff_ast = DiffAST.from_json(obj4, id_to_type)
    diff_ast = None
    return Example(obj1['id'], obj1['label'], old_comment_subtokens,
                   span_diff_code_subtokens, None, None, diff_ast, None, None)


@enum.unique
class DiffEdgeType(Enum):
    PARENT = 0
    CHILD = 1
    PREV_SIBLING = 2
    NEXT_SIBLING = 3
    ALIGNED_NEIGHBOR = 4


def insert_graph(batch, ast, node_features, vocabulary):
    batch.root_ids.append(batch.num_nodes)  # list of int
    graph_node_positions = []
    for n, node in enumerate(ast.nodes):
        batch.graph_ids.append(batch.num_graphs)
        batch.is_internal.append(not node.is_leaf)
        # 这里之前是一个list,现在应该是list of list
        # batch.value_lookup_ids.append(vocabulary.get_id_or_unk(node.value))
        sub_ids = [vocabulary.get_id_or_unk(node.attribute)] + vocabulary.get_id_or_unk_multiple(node.subtokens, MAX_SUBTOKENS-1)
        batch.value_lookup_ids.append(sub_ids)

        if node.action_type == 'Insert':
            src_type = SrcType.INSERT
        elif node.action_type == 'Delete':
            src_type = SrcType.DELETE
        elif node.action_type == 'Move':
            src_type = SrcType.MOVE
        elif node.src == 'old' and node.action_type == 'Update':
            src_type = SrcType.REPLACE_OLD
        elif node.src == 'new' and node.action_type == 'Update':
            src_type = SrcType.REPLACE_NEW
        else:
            src_type = SrcType.KEEP

        batch.src_type_ids.append(src_type.value)
        graph_node_positions.append(batch.num_nodes + node.node_id)  # 本身id加上偏移量

        for parent in node.parents:
            if parent.node_id < len(ast.nodes):
                batch.edges[DiffEdgeType.PARENT.value].append(
                    (batch.num_nodes + node.node_id, batch.num_nodes + parent.node_id))

        for child in node.children:
            if child.node_id < len(ast.nodes):
                batch.edges[DiffEdgeType.CHILD.value].append(
                    (batch.num_nodes + node.node_id, batch.num_nodes + child.node_id))

        for next_sibling in node.next_siblings:
            if next_sibling.node_id < len(ast.nodes):
                batch.edges[DiffEdgeType.NEXT_SIBLING.value].append(
                    (batch.num_nodes + node.node_id, batch.num_nodes + next_sibling.node_id))

        for prev_sibling in node.prev_siblings:
            if prev_sibling.node_id < len(ast.nodes):
                batch.edges[DiffEdgeType.PREV_SIBLING.value].append(
                    (batch.num_nodes + node.node_id, batch.num_nodes + prev_sibling.node_id))

        # for subtoken_parent in node.subtoken_parents:
        #     if subtoken_parent.node_id < len(ast.nodes):
        #         batch.edges[DiffEdgeType.SUBTOKEN_PARENT.value].append(
        #             (batch.num_nodes + node.node_id, batch.num_nodes + subtoken_parent.node_id))

        # for subtoken_child in node.subtoken_children:
        #     if subtoken_child.node_id < len(ast.nodes):
        #         batch.edges[DiffEdgeType.SUBTOKEN_CHILD.value].append(
        #             (batch.num_nodes + node.node_id, batch.num_nodes + subtoken_child.node_id))

        # for next_subtoken in node.next_subtokens:
        #     if next_subtoken.node_id < len(ast.nodes):
        #         batch.edges[DiffEdgeType.NEXT_SUBTOKEN.value].append(
        #             (batch.num_nodes + node.node_id, batch.num_nodes + next_subtoken.node_id))

        # for prev_subtoken in node.prev_subtokens:
        #     if prev_subtoken.node_id < len(ast.nodes):
        #         batch.edges[DiffEdgeType.PREV_SUBTOKEN.value].append(
        #             (batch.num_nodes + node.node_id, batch.num_nodes + prev_subtoken.node_id))

        for aligned_neighbor in node.aligned_neighbors:
            if aligned_neighbor.node_id < len(ast.nodes):
                batch.edges[DiffEdgeType.ALIGNED_NEIGHBOR.value].append(
                    (batch.num_nodes + node.node_id, batch.num_nodes + aligned_neighbor.node_id))

    if node_features is not None:
        batch.node_features.extend(node_features)
    batch.node_positions.append(graph_node_positions)  # list of list
    batch.num_nodes_per_graph.append(len(ast.nodes))  # list of int
    batch.num_nodes += len(ast.nodes)  # int
    batch.num_graphs += 1  # int
    return batch


class UpdateBatchData(NamedTuple):
    """Stores tensorized batch used in edit model."""
    code_ids: torch.Tensor
    code_lengths: torch.Tensor
    old_nl_ids: torch.Tensor
    old_nl_lengths: torch.Tensor
    old_nl_start: torch.Tensor
    old_nl_end: torch.Tensor
    trg_nl_ids: torch.Tensor
    trg_extended_nl_ids: torch.Tensor
    trg_nl_lengths: torch.Tensor
    invalid_copy_positions: torch.Tensor
    input_str_reps: List[List[str]]
    input_ids: List[List[str]]
    code_features: torch.Tensor
    nl_features: torch.Tensor
    labels: torch.Tensor
    # for scripts
    action_ids: torch.Tensor
    old_value_ids: torch.Tensor
    new_value_ids: torch.Tensor
    old_path_ids: torch.Tensor
    new_path_ids: torch.Tensor

    graph_batch: GraphMethodBatch
    nl_code_edges: ndarray


actions = {
    'Insert': 1,
    'Delete': 2,
    'Update': 3,
    'Move': 5,
}


def tensorize_graph_method_batch(batch, device, max_num_nodes_per_graph):
    node_positions = np.zeros([batch.num_graphs, max_num_nodes_per_graph], dtype=np.int32)
    for g in range(batch.num_graphs):
        graph_node_positions = batch.node_positions[g]
        node_positions[g, :len(graph_node_positions)] = graph_node_positions
        node_positions[g, len(graph_node_positions):] = batch.root_ids[g]  # 这个是填充

    return GraphMethodBatch(
        torch.tensor(batch.graph_ids, dtype=torch.int32, device=device),
        torch.tensor(batch.value_lookup_ids, dtype=torch.int32, device=device),
        torch.tensor(batch.src_type_ids, dtype=torch.int32, device=device),
        torch.tensor(batch.root_ids, dtype=torch.int32, device=device),
        torch.tensor(batch.is_internal, dtype=torch.uint8, device=device),
        batch.edges, batch.num_graphs, batch.num_nodes,
        torch.tensor(batch.node_features, dtype=torch.float32, device=device),
        torch.tensor(node_positions, dtype=torch.int32, device=device),
        torch.tensor(batch.num_nodes_per_graph, dtype=torch.int32, device=device))


def build_batch(batch_examples: List[Example], embedding_store: EmbeddingStore,
                max_nl_length, max_code_length, max_ast_length, device,
                ):
    code_token_ids = []
    code_lengths = []
    old_nl_token_ids = []
    old_nl_lengths = []
    old_nl_start = []
    old_nl_end = []
    # trg_token_ids = []
    # trg_extended_token_ids = []
    # trg_lengths = []
    # invalid_copy_positions = []
    # inp_str_reps = []
    # inp_ids = []
    code_features = []
    nl_features = []
    labels = []
    nl_code_edges = []
    batch_ids = []
    action_ids = []
    old_value_ids = []
    new_value_ids = []
    old_path_ids = []
    new_path_ids = []
    graph_batch = initialize_graph_method_batch(len(DiffEdgeType))
    for i, ex in enumerate(batch_examples):
        code_sequence_ids = embedding_store.get_padded_code_ids(
            ex.span_diff_code_subtokens, max_code_length)
        code_length = min(len(ex.span_diff_code_subtokens), max_code_length)
        code_token_ids.append(code_sequence_ids)
        code_lengths.append(code_length)

        if max_ast_length > 0:  # 只有设置了max_ast_length才会处理ast
            graph_batch = insert_graph(graph_batch, ex.diff_ast, ex.node_features, embedding_store.code_vocabulary)

        old_nl_sequence = ex.old_context_subtokens
        old_nl_length = min(len(old_nl_sequence), max_nl_length)
        old_nl_sequence_ids = embedding_store.get_padded_nl_ids(
            old_nl_sequence, max_nl_length)
        old_nl_token_ids.append(old_nl_sequence_ids)
        old_nl_lengths.append(old_nl_length)
        old_nl_start.append(min(ex.nl_start, old_nl_length-1))
        old_nl_end.append(min(ex.nl_end, old_nl_length))

        # TODO: generate task

        labels.append(ex.label)
        code_features.append(ex.code_feature)
        nl_features.append(ex.nl_feature)
        batch_ids += [i] * len(ex.token_match)
        nl_code_edges.extend(ex.token_match)

        action_id = []
        old_value_id = []
        new_value_id = []
        old_path_id = []
        new_path_id = []
        for script in ex.diff_ast.scripts[:MAX_SCRIPT_LENGTH]:
            action_id.append(actions.get(script.action, 0))
            old_value_id.append(embedding_store.get_padded_code_ids(script.old_value, MAX_SUBTOKENS))
            new_value_id.append(embedding_store.get_padded_code_ids(script.new_value, MAX_SUBTOKENS))
            old_path_id.append(embedding_store.get_padded_code_ids(script.old_path, MAX_SUBTOKENS))
            new_path_id.append(embedding_store.get_padded_code_ids(script.new_path, MAX_SUBTOKENS))
        pad_length = MAX_SCRIPT_LENGTH - len(action_id)
        if pad_length > 0:
            action_id += [0] * pad_length
            old_value_id += [[0] * MAX_SUBTOKENS] * pad_length
            new_value_id += [[0] * MAX_SUBTOKENS] * pad_length
            old_path_id += [[0] * MAX_SUBTOKENS] * pad_length
            new_path_id += [[0] * MAX_SUBTOKENS] * pad_length
        action_ids.append(action_id)
        old_value_ids.append(old_value_id)
        new_value_ids.append(new_value_id)
        old_path_ids.append(old_path_id)
        new_path_ids.append(new_path_id)

    edges = None
    if max_ast_length > 0:
        edges = np.array(nl_code_edges)
        edge_type = edges[:, -1]
        edge_idxs = np.array(batch_ids) * max_nl_length * max_ast_length
        edge_idxs += (edges[:, 0] * max_ast_length + edges[:, 1])
        edges = np.column_stack((edge_idxs, edge_type))

    return UpdateBatchData(
        torch.tensor(code_token_ids, dtype=torch.int32, device=device),
        torch.tensor(code_lengths, dtype=torch.int32, device=device),
        torch.tensor(old_nl_token_ids, dtype=torch.int32, device=device),
        torch.tensor(old_nl_lengths, dtype=torch.int32, device=device),
        torch.tensor(old_nl_start, dtype=torch.int32, device=device),
        torch.tensor(old_nl_end, dtype=torch.int32, device=device),
        None,
        None,
        None,
        None,
        None,
        None,
        torch.tensor(code_features, dtype=torch.float32, device=device),
        torch.tensor(nl_features, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.bool, device=device),
        torch.tensor(action_ids, dtype=torch.int32, device=device),
        torch.tensor(old_value_ids, dtype=torch.int32, device=device),
        torch.tensor(new_value_ids, dtype=torch.int32, device=device),
        torch.tensor(old_path_ids, dtype=torch.int32, device=device),
        torch.tensor(new_path_ids, dtype=torch.int32, device=device),
        tensorize_graph_method_batch(graph_batch, device, max_ast_length),
        edges,
    )


def batch_generator(examples, batch_size, embedding_store: EmbeddingStore,
                    max_nl_length, max_code_length, max_ast_length, device,
                    shuffle=False):
    batch_examples = []
    example_count = 0
    for example in examples:
        batch_examples.append(example)
        example_count += 1
        if example_count == batch_size:
            yield build_batch(batch_examples, embedding_store, max_nl_length, max_code_length, max_ast_length, device)
            batch_examples = []
            example_count = 0
    if example_count > 0:
        yield build_batch(batch_examples, embedding_store, max_nl_length, max_code_length, max_ast_length, device)


def build_features(seq_obj, max_nl_length, max_code_length):
    old_nl_sequence = seq_obj['ocs']
    frequency_map = defaultdict(int)
    for tok in old_nl_sequence:
        frequency_map[tok] += 1
    pos_tags = pos_tag(old_nl_sequence)
    pos_tag_indices = []
    for _, t in pos_tags:
        if t in tags:
            pos_tag_indices.append(tags.index(t))
        else:
            pos_tag_indices.append(tags.index('OTHER'))
    ortt = set(seq_obj['ortt'])
    nrtt = set(seq_obj['nrtt'])
    rti = ortt & nrtt
    orlt = set(seq_obj['orlt'])
    nrlt = set(seq_obj['nrlt'])
    rli = orlt & nrlt
    ict = set(seq_obj['ict'])
    kct = set(seq_obj['kct'])
    dct = set(seq_obj['dct'])
    roct = set(seq_obj['roct'])
    rnct = set(seq_obj['rnct'])
    nl_features = np.zeros((max_nl_length, get_num_nl_features()), dtype=np.int32)
    for i, token in enumerate(old_nl_sequence):
        if i >= max_nl_length:
            break
        if token in rti:
            nl_features[i, 0] = 1
        elif token in ortt:
            nl_features[i, 1] = 1
        elif token in nrtt:
            nl_features[i, 2] = 1
        else:
            nl_features[i, 3] = 1
        if token in rli:
            nl_features[i, 4] = 1
        elif token in orlt:
            nl_features[i, 5] = 1
        elif token in nrlt:
            nl_features[i, 6] = 1
        else:
            nl_features[i, 7] = 1
        nl_features[i, 8] = token in ict
        nl_features[i, 9] = token in kct
        nl_features[i, 10] = token in dct
        nl_features[i, 11] = token in roct
        nl_features[i, 12] = token in rnct
        nl_features[i, 13] = token in stop_words
        nl_features[i, 14] = frequency_map[token] > 1
        nl_features[i, 15] = seq_obj['ocsl'][i]
        nl_features[i, 16] = seq_obj['ocsi'][i]
        nl_features[i, 17+pos_tag_indices[i]] = 1

    code_features = np.zeros((max_code_length, get_num_code_features()), dtype=np.int32)
    old_nl_tokens = set(old_nl_sequence)
    last_command = None
    for i, token in enumerate(seq_obj['sdcs']):
        if i >= max_code_length:
            break
        if token in rti:
            code_features[i, 0] = 1
        elif token in ortt:
            code_features[i, 1] = 1
        elif token in nrtt:
            code_features[i, 2] = 1
        else:
            code_features[i, 3] = 1
        if token in rli:
            code_features[i, 4] = 1
        elif token in orlt:
            code_features[i, 5] = 1
        elif token in nrlt:
            code_features[i, 6] = 1
        else:
            code_features[i, 7] = 1
        if is_edit_keyword(token):
            code_features[i, 8] = 1
        if is_java_keyword(token):
            code_features[i, 9] = 1
        if is_operator(token):
            code_features[i, 10] = 1
        if token in old_nl_tokens:
            code_features[i, 11] = 1
        if not is_edit_keyword(token):
            if last_command == KEEP:
                code_features[i, 12] = 1
            elif last_command == INSERT:
                code_features[i, 13] = 1
            elif last_command == DELETE:
                code_features[i, 14] = 1
            elif last_command == REPLACE_NEW:
                code_features[i, 15] = 1
            else:
                code_features[i, 16] = 1
        else:
            last_command = token
        code_features[i, 17] = seq_obj['essl'][i]
        code_features[i, 18] = seq_obj['essi'][i]
    return nl_features.astype(np.float32), code_features.astype(np.float32)


def build_node_features(nodes, obj, max_nl_length):
    # 因为传递的nodes的数量已经截断,所以不需要传递max_ast_length
    # max_nl_length主要用于限制old_nl_subtoken_map的长度
    # (
    #     old_return_line_terms, new_return_line_terms, return_line_intersection,
    #     old_set, new_set, intersection
    # ) = detail_tmp
    lem = Lemmatizer()
    features = np.zeros((len(nodes), get_num_node_features()), dtype=np.float32)

    # old_nl_tokens = obj['old_comment_subtokens']

    subtoken_positions = defaultdict(list)
    sublemma_positions = defaultdict(list)
    for i, subtoken in enumerate(obj['ocs'][:max_nl_length]):
        if subtoken.isalnum():
            subtoken_positions[subtoken].append(i)
            lemma = lem.lemmatize(subtoken)
            if lemma != subtoken:
                sublemma_positions[lemma].append(i)
    old_nl_subtokens = set(subtoken_positions.keys())
    lemmas = set(sublemma_positions.keys())

    token_positions = defaultdict(list)
    for i, token in enumerate(obj["octm"][:max_nl_length]):
        token_positions[token].append(i)

    token_match = []  # [[1, 5, 0], [3, 5, 0]] for example, [nl_pos, node_pos, edge_type]
    nl_match_positions = set()
    for i, node in enumerate(nodes):
        if not node.is_leaf:
            features[i][-1] = 1.0
            continue

        token = node.value  # 子词经过了lower,但是Token没有
        # node.value = token.lower() # 在这里将value转为小写,完成DiffAST.from_json 的工作
        if token in token_positions:
            features[i][0] = 1.0
            for j in token_positions[token]:
                token_match.append([j, i, 0])  # [nl_pos, node_pos, edge_type]
                nl_match_positions.add(j)
        else:
            sub_len = max(len(node.subtokens), 1)  # divide by 0
            shared_subtokens = old_nl_subtokens & set(node.subtokens)
            features[i][1] = len(shared_subtokens) / sub_len
            for subtoken in shared_subtokens:
                for j in subtoken_positions[subtoken]:
                    token_match.append([j, i, 1])
            shared_lemmas = lemmas & set(node.subtokens)
            features[i][2] = len(shared_lemmas) / sub_len
            for lemma in shared_lemmas:
                for j in sublemma_positions[lemma]:
                    token_match.append([j, i, 2])

        # if token in intersection:
        #     features[i][0] = True
        # elif token in old_set:
        #     features[i][1] = True
        # elif token in new_set:
        #     features[i][2] = True
        # else:
        #     features[i][3] = True

        # if token in return_line_intersection:
        #     features[i][4] = True
        # elif token in old_return_line_terms:
        #     features[i][5] = True
        # elif token in new_return_line_terms:
        #     features[i][6] = True
        # else:
        #     features[i][7] = True

        # if is_edit_keyword(token):
        #     features[i][8] = True
        # if is_java_keyword(token):
        #     features[i][9] = True
        # if is_operator(token):
        #     features[i][10] = True
        # if token in old_nl_tokens:
        #     features[i][11] = True

        # if len(node.subtoken_children) > 0 or len(node.subtoken_parents) > 0:
        #     features[i][17] = True

        # if len(node.subtoken_parents) == 1:
        #     features[i][18] = node.subtoken_parents[0].subtoken_children.index(node)

    # return features.astype(np.float32), token_match
    # 这里为了限制edge数量,进行截取,防止out of memory
    token_match = [i for i in token_match if i[0] not in nl_match_positions or i[2] == 0][:max_nl_length*2]
    return features, token_match


def build_example(seq_obj, max_nl_length, max_code_length,
                  tree_obj=None, max_ast_length=0, id_to_type=None):  # 这三个参数是为AST数据准备的
    nl_feature, code_feature = build_features(seq_obj, max_nl_length, max_code_length)

    # 处理AST数据
    diff_ast = None
    node_features = None
    token_match = None
    if max_ast_length != 0:
        diff_ast = DiffAST.from_json(tree_obj, id_to_type)
        diff_ast.nodes = diff_ast.nodes[:max_ast_length]  # 这里使用到了max_ast_length
        # nodes和node_features的长度一致,都不大于max_ast_length
        node_features, token_match = build_node_features(diff_ast.nodes, seq_obj, max_nl_length)

    return Example(seq_obj['id'], seq_obj['label'], seq_obj['ocs'], seq_obj['s_e'][0], seq_obj['s_e'][1],
                   seq_obj['sdcs'], nl_feature, code_feature, diff_ast, node_features,
                   token_match)


def example_generator(files, max_nl_length, max_code_length, max_ast_length):
    """
    files: [seq, tree]
    """
    with open('/data/share/kingxu/data/CUP/ast_type_dict.json') as f:
        type_to_id = json.load(f)
    id_to_type = {v: k for k, v in type_to_id.items()}
    while True:
        lines = [f.readline() for f in files]
        line = lines[0].strip()
        if not line:
            break
        ex = build_example(json.loads(line), max_nl_length, max_code_length,
                           json.loads(lines[1]), max_ast_length, id_to_type)
        yield ex


def get_opened_files(dir, type):
    """
    与close_files()成对使用
    """
    files = [
        '{}_seq.jsonl',
        '{}_tree.jsonl',
    ]
    dir = Path(dir)
    files = [dir / f.format(type) for f in files]
    fs = [open(f, 'r') for f in files]
    return fs


def close_files(files):
    for f in files:
        f.close()


def load_batch_data_to_cpu(dir, type, max_nl_length, max_code_length, max_ast_length, batch_size, embedding_store):
    batches = []
    device = torch.device('cpu')
    fs = get_opened_files(dir, type)
    examples = example_generator(fs, max_nl_length, max_code_length, max_ast_length)
    # 重新修改了 build_batch 和 tensorize_graph_method_batch 其中int以int32保存,减少内存占用,移到GPU时进行类型转换
    count = 0
    for batch in tqdm(batch_generator(examples, batch_size, embedding_store, max_nl_length, max_code_length, max_ast_length, device)):
        batches.append(batch)
        count += 1
        if count > 6000:
            break
    close_files(fs)
    return batches


def test():
    dir = Path('/mnt/data1/kingxu/data/JIT_ID/data/resub_cup2')
    fs = get_opened_files(dir, 'valid')
    # examples = list(example_generator(fs, max_nl_length=25, max_code_length=380))
    examples = example_generator(fs, max_nl_length=200, max_code_length=380, max_ast_length=300)

    from dpu_utils.mlutils import Vocabulary
    with open('/mnt/data1/kingxu/data/JIT_ID/data/cup2_mix_re_subword_vocab.json', 'r') as f:
        obj = json.load(f)
    max_nl_length = 200
    nl_vocab = Vocabulary()
    nl_vocab.token_to_id = obj['nl_vocab']['token_to_id']
    nl_vocab.id_to_token = obj['nl_vocab']['id_to_token']
    # code_vocab = Vocabulary()
    # code_vocab.token_to_id = obj['code_vocab']['token_to_id']
    # code_vocab.id_to_token = obj['code_vocab']['id_to_token']
    embedding_store = EmbeddingStore(nl_vocab, 60, 0.1, 6, 60, 60, False)
    # for ex in examples:
    # print(ex)
    # print(ex.nl_feature.shape)
    # print(ex.code_feature.shape)
    # break
    device = torch.device('cpu')
    batches = batch_generator(examples, batch_size=32, embedding_store=embedding_store,
                              max_nl_length=max_nl_length, max_code_length=380, max_ast_length=300, device=device)
    for batch in batches:
        print(batch)
        # print(batch.graph_batch.__dict__)
        break
    close_files(fs)


def test_DiffAST():
    file = '/mnt/data1/kingxu/data/CUP/cup2_dataset/test_tree.jsonl'
    type_to_id = json.load(open('/mnt/data1/kingxu/data/CUP/cup2_dataset/ast_type_dict.json'))
    id_to_type = {v: k for k, v in type_to_id.items()}
    with open(file, 'r') as f:
        for l in tqdm(f):
            obj = json.loads(l)
            diff_ast = DiffAST.from_json(obj, id_to_type)
            print(diff_ast.scripts)
            # print(diff_ast.to_json())


def save_positive_examples(dir):
    dir = Path(dir)
    files = [
        '{}_nl_enhenced.jsonl',
        '{}_high_level_details.jsonl',
        '{}_tokenization_features.jsonl',
        '{}_tree.jsonl',
        '{}.jsonl',
    ]
    in_fs = [open(dir / f.format('train')) for f in files]
    out_fs = [open(dir / f.format('p_train'), 'w') for f in files]
    while True:
        lines = [f.readline() for f in in_fs]
        line1 = lines[0].strip()
        if not line1:
            break
        obj = json.loads(line1)
        if obj['label']:
            for i in range(len(files)):
                out_fs[i].write(lines[i])
    [f.close() for f in in_fs]
    [f.close() for f in out_fs]


if __name__ == "__main__":
    # test()
    test_DiffAST()
    # save_positive_examples('/mnt/data1/kingxu/data/JIT_ID/data/resub_cup2')
