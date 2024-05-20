import json
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from data_utils import (DiffEdgeType, Example, GraphMethodBatch, SrcType,
                        build_example, initialize_graph_method_batch)
from embedding_store import EmbeddingStore


class MyDataset(Dataset):
    def __init__(self, dir: Path, type, max_nl_length, max_code_length, max_ast_length=0, id_to_type=None, subset='a'):
        """ subset:'a','r','u'; 'a' for all,'r' for reliable set,'u' for unreliable set """
        super(MyDataset).__init__()
        self.mnl = max_nl_length
        self.mcl = max_code_length
        self.mal = max_ast_length
        self.itt = id_to_type

        self.seq = []
        self.tree = []

        files = ['{}_seq.jsonl', '{}_tree.jsonl']
        seq, tree = [], []
        print(f'Loading {type} seq ...', flush=True)
        with open(dir / files[0].format(type)) as f:
            for l in f:
                seq.append(l)
        print(f"Loading {type} tree ...", flush=True)
        with open(dir / files[1].format(type)) as f:
            for l in f:
                tree.append(l)
        assert len(seq) == len(tree)
        for l1, l2 in zip(seq, tree):
            if l1.split('"cf": ', 1)[1][0] == '1':  # sample is reliable
                if subset == 'u':  # unreliable needed
                    continue
            else:  # sample is unreliable
                if subset == 'r':  # reliable needed
                    continue
            self.seq.append(l1)
            self.tree.append(l2)
        del seq, tree  # free space
        print(f'Loading {len(self.seq)} done!', flush=True)

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, i: int) -> Example:
        return build_example(json.loads(self.seq[i]), self.mnl, self.mcl,
                             json.loads(self.tree[i]), self.mal, self.itt)

    def append(self, dataset):
        self.seq += dataset.seq
        self.tree += dataset.tree


def insert_graph(batch, ast, node_features, vocabulary, max_subtokens):
    batch.root_ids.append(batch.num_nodes)  # list of int
    graph_node_positions = []
    for n, node in enumerate(ast.nodes):
        batch.graph_ids.append(batch.num_graphs)
        batch.is_internal.append(not node.is_leaf)
        # batch.value_lookup_ids.append(vocabulary.get_id_or_unk(node.value))
        # list of list
        sub_ids = [vocabulary.get_id_or_unk(node.attribute)] + vocabulary.get_id_or_unk_multiple(node.subtokens, max_subtokens-1)
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
        graph_node_positions.append(batch.num_nodes + node.node_id)  # start + offset

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


class UpdateBatchData:
    """Stores tensorized batch used in edit model."""

    def __init__(self, code_ids, code_lengths, old_nl_ids, old_nl_lengths,
                 old_nl_start, old_nl_end, code_features, nl_features, labels,
                 action_ids, old_value_ids, new_value_ids, old_path_ids, new_path_ids,
                 nl_code_edges, graph_batch) -> None:
        self.code_ids = code_ids
        self.code_lengths = code_lengths
        self.old_nl_ids = old_nl_ids
        self.old_nl_lengths = old_nl_lengths
        self.old_nl_start = old_nl_start
        self.old_nl_end = old_nl_end
        self.code_features = code_features
        self.nl_features = nl_features
        self.labels = labels
        self.action_ids = action_ids
        self.old_value_ids = old_value_ids
        self.new_value_ids = new_value_ids
        self.old_path_ids = old_path_ids
        self.new_path_ids = new_path_ids
        self.nl_code_edges = nl_code_edges
        self.graph_batch = graph_batch

    def pin_memory(self):
        # graph_batch = GraphMethodBatch(
        #     self.graph_batch.graph_ids.pin_memory(),
        #     self.graph_batch.value_lookup_ids.pin_memory(),
        #     self.graph_batch.src_type_ids.pin_memory(),
        #     self.graph_batch.root_ids.pin_memory(),
        #     self.graph_batch.is_internal.pin_memory(),
        #     self.graph_batch.edges,
        #     self.graph_batch.num_graphs,
        #     self.graph_batch.num_nodes,
        #     self.graph_batch.node_features.pin_memory(),
        #     self.graph_batch.node_positions.pin_memory(),
        #     self.graph_batch.num_nodes_per_graph.pin_memory()
        # )
        # return UpdateBatchData(
        #     self.code_ids.pin_memory(),
        #     self.code_lengths.pin_memory(),
        #     self.old_nl_ids.pin_memory(),
        #     self.old_nl_lengths.pin_memory(),
        #     self.old_nl_start.pin_memory(),
        #     self.old_nl_end.pin_memory(),
        #     self.code_features.pin_memory(),
        #     self.nl_features.pin_memory(),
        #     self.labels.pin_memory(),
        #     self.action_ids.pin_memory(),
        #     self.old_value_ids.pin_memory(),
        #     self.new_value_ids.pin_memory(),
        #     self.old_path_ids.pin_memory(),
        #     self.new_path_ids.pin_memory(),
        #     self.nl_code_edges.pin_memory(),
        #     graph_batch,
        # )
        self.code_ids = self.code_ids.pin_memory()
        self.code_lengths = self.code_lengths.pin_memory()
        self.old_nl_ids = self.old_nl_ids.pin_memory()
        self.old_nl_lengths = self.old_nl_lengths.pin_memory()
        self.old_nl_start = self.old_nl_start.pin_memory()
        self.old_nl_end = self.old_nl_end.pin_memory()
        self.code_features = self.code_features.pin_memory()
        self.nl_features = self.nl_features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.action_ids = self.action_ids.pin_memory()
        self.old_value_ids = self.old_value_ids.pin_memory()
        self.new_value_ids = self.new_value_ids.pin_memory()
        self.old_path_ids = self.old_path_ids.pin_memory()
        self.new_path_ids = self.new_path_ids.pin_memory()
        self.nl_code_edges = self.nl_code_edges.pin_memory()
        self.graph_batch.graph_ids = self.graph_batch.graph_ids.pin_memory()
        self.graph_batch.value_lookup_ids = self.graph_batch.value_lookup_ids.pin_memory()
        self.graph_batch.src_type_ids = self.graph_batch.src_type_ids.pin_memory()
        self.graph_batch.root_ids = self.graph_batch.root_ids.pin_memory()
        self.graph_batch.is_internal = self.graph_batch.is_internal.pin_memory()
        for i in range(len(self.graph_batch.edges)):
            self.graph_batch.edges[i] = self.graph_batch.edges[i].pin_memory()
        self.graph_batch.num_graphs = self.graph_batch.num_graphs.pin_memory()
        self.graph_batch.num_nodes = self.graph_batch.num_nodes.pin_memory()
        self.graph_batch.node_features = self.graph_batch.node_features.pin_memory()
        self.graph_batch.node_positions = self.graph_batch.node_positions.pin_memory()
        self.graph_batch.num_nodes_per_graph = self.graph_batch.num_nodes_per_graph.pin_memory()
        return self

    # def to_device(self, device):
    #     self.code_ids = self.code_ids.to(device=device, non_blocking=True)
    #     self.code_lengths = self.code_lengths.to(device=device, non_blocking=True)
    #     self.old_nl_ids = self.old_nl_ids.to(device=device, non_blocking=True)
    #     self.old_nl_lengths = self.old_nl_lengths.to(device=device, non_blocking=True)
    #     self.old_nl_start = self.old_nl_start.to(device=device, non_blocking=True)
    #     self.old_nl_end = self.old_nl_end.to(device=device, non_blocking=True)
    #     self.code_features = self.code_features.to(device=device, non_blocking=True)
    #     self.nl_features = self.nl_features.to(device=device, non_blocking=True)
    #     self.labels = self.labels.to(device=device, non_blocking=True)
    #     self.action_ids = self.action_ids.to(device=device, non_blocking=True)
    #     self.old_value_ids = self.old_value_ids.to(device=device, non_blocking=True)
    #     self.new_value_ids = self.new_value_ids.to(device=device, non_blocking=True)
    #     self.old_path_ids = self.old_path_ids.to(device=device, non_blocking=True)
    #     self.new_path_ids = self.new_path_ids.to(device=device, non_blocking=True)
    #     self.nl_code_edges = self.nl_code_edges.to(device=device, non_blocking=True)
    #     self.graph_batch.graph_ids = self.graph_batch.graph_ids.to(device=device, non_blocking=True)
    #     self.graph_batch.value_lookup_ids = self.graph_batch.value_lookup_ids.to(device=device, non_blocking=True)
    #     self.graph_batch.src_type_ids = self.graph_batch.src_type_ids.to(device=device, non_blocking=True)
    #     self.graph_batch.root_ids = self.graph_batch.root_ids.to(device=device, non_blocking=True)
    #     self.graph_batch.is_internal = self.graph_batch.is_internal.to(device=device, non_blocking=True)
    #     self.graph_batch.node_features = self.graph_batch.node_features.to(device=device, non_blocking=True)
    #     self.graph_batch.node_positions = self.graph_batch.node_positions.to(device=device, non_blocking=True)
    #     self.graph_batch.num_nodes_per_graph = self.graph_batch.num_nodes_per_graph.to(device=device, non_blocking=True)
    #     return self

    def to_device_new_obj(self, device):
        graph_batch = GraphMethodBatch(
            self.graph_batch.graph_ids.to(device=device, non_blocking=True),
            self.graph_batch.value_lookup_ids.to(device=device, non_blocking=True),
            self.graph_batch.src_type_ids.to(device=device, non_blocking=True),
            self.graph_batch.root_ids.to(device=device, non_blocking=True),
            self.graph_batch.is_internal.to(device=device, non_blocking=True),
            [t.to(device=device, non_blocking=True) for t in self.graph_batch.edges],
            self.graph_batch.num_graphs.to(device=device, non_blocking=True),
            self.graph_batch.num_nodes.to(device=device, non_blocking=True),
            self.graph_batch.node_features.to(device=device, non_blocking=True),
            self.graph_batch.node_positions.to(device=device, non_blocking=True),
            self.graph_batch.num_nodes_per_graph.to(device=device, non_blocking=True)
        )
        return UpdateBatchData(
            self.code_ids.to(device=device, non_blocking=True),
            self.code_lengths.to(device=device, non_blocking=True),
            self.old_nl_ids.to(device=device, non_blocking=True),
            self.old_nl_lengths.to(device=device, non_blocking=True),
            self.old_nl_start.to(device=device, non_blocking=True),
            self.old_nl_end.to(device=device, non_blocking=True),
            self.code_features.to(device=device, non_blocking=True),
            self.nl_features.to(device=device, non_blocking=True),
            self.labels.to(device=device, non_blocking=True),
            self.action_ids.to(device=device, non_blocking=True),
            self.old_value_ids.to(device=device, non_blocking=True),
            self.new_value_ids.to(device=device, non_blocking=True),
            self.old_path_ids.to(device=device, non_blocking=True),
            self.new_path_ids.to(device=device, non_blocking=True),
            self.nl_code_edges.to(device=device, non_blocking=True),
            graph_batch,
        )


def tensorize_graph_method_batch(batch, max_num_nodes_per_graph):
    node_positions = np.zeros([batch.num_graphs, max_num_nodes_per_graph], dtype=np.int64)
    for g in range(batch.num_graphs):
        graph_node_positions = batch.node_positions[g]
        node_positions[g, :len(graph_node_positions)] = graph_node_positions
        node_positions[g, len(graph_node_positions):] = batch.root_ids[g]  # 这个是填充
    edges = []
    for e in batch.edges:
        edges.append(torch.tensor(e, dtype=torch.int64))
    return GraphMethodBatch(
        torch.tensor(batch.graph_ids, dtype=torch.int64),
        torch.tensor(batch.value_lookup_ids, dtype=torch.int64),
        torch.tensor(batch.src_type_ids, dtype=torch.int64),
        torch.tensor(batch.root_ids, dtype=torch.int64),
        torch.tensor(batch.is_internal, dtype=torch.uint8),
        edges,
        torch.tensor(batch.num_graphs),
        torch.tensor(batch.num_nodes),
        torch.tensor(batch.node_features, dtype=torch.float32),
        torch.tensor(node_positions, dtype=torch.int64),
        torch.tensor(batch.num_nodes_per_graph, dtype=torch.int64))


def build_batch(batch_examples: List[Example], embedding_store: EmbeddingStore,
                max_nl_length, max_code_length, max_ast_length, max_script_length,
                max_subtokens):
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

    actions = {
        'Insert': 1,
        'Delete': 2,
        'Update': 3,
        'Move': 5,
    }

    graph_batch = initialize_graph_method_batch(len(DiffEdgeType))
    for i, ex in enumerate(batch_examples):
        code_sequence_ids = embedding_store.get_padded_code_ids(
            ex.span_diff_code_subtokens, max_code_length)
        code_length = min(len(ex.span_diff_code_subtokens), max_code_length)
        code_token_ids.append(code_sequence_ids)
        code_lengths.append(code_length)

        if max_ast_length > 0:  # 只有设置了max_ast_length才会处理ast
            graph_batch = insert_graph(graph_batch, ex.diff_ast, ex.node_features,
                                       embedding_store.code_vocabulary, max_subtokens)

        old_nl_sequence = ex.old_context_subtokens
        old_nl_length = min(len(old_nl_sequence), max_nl_length)
        old_nl_sequence_ids = embedding_store.get_padded_nl_ids(
            old_nl_sequence, max_nl_length)
        old_nl_token_ids.append(old_nl_sequence_ids)
        old_nl_lengths.append(old_nl_length)
        old_nl_start.append(min(ex.nl_start, old_nl_length-1))
        old_nl_end.append(min(ex.nl_end, old_nl_length))

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
        for script in ex.diff_ast.scripts[:max_script_length]:
            action_id.append(actions.get(script.action, 0))
            old_value_id.append(embedding_store.get_padded_code_ids(script.old_value, max_subtokens))
            new_value_id.append(embedding_store.get_padded_code_ids(script.new_value, max_subtokens))
            old_path_id.append(embedding_store.get_padded_code_ids(script.old_path, max_subtokens))
            new_path_id.append(embedding_store.get_padded_code_ids(script.new_path, max_subtokens))
        pad_length = max_script_length - len(action_id)
        if pad_length > 0:
            action_id += [0] * pad_length
            old_value_id += [[0] * max_subtokens] * pad_length
            new_value_id += [[0] * max_subtokens] * pad_length
            old_path_id += [[0] * max_subtokens] * pad_length
            new_path_id += [[0] * max_subtokens] * pad_length
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
        torch.tensor(code_token_ids, dtype=torch.int64),
        torch.tensor(code_lengths, dtype=torch.int64),
        torch.tensor(old_nl_token_ids, dtype=torch.int64),
        torch.tensor(old_nl_lengths, dtype=torch.int64),
        torch.tensor(old_nl_start, dtype=torch.int64),
        torch.tensor(old_nl_end, dtype=torch.int64),
        torch.tensor(code_features, dtype=torch.float32),
        torch.tensor(nl_features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.bool),
        torch.tensor(action_ids, dtype=torch.int64),
        torch.tensor(old_value_ids, dtype=torch.int64),
        torch.tensor(new_value_ids, dtype=torch.int64),
        torch.tensor(old_path_ids, dtype=torch.int64),
        torch.tensor(new_path_ids, dtype=torch.int64),
        torch.tensor(edges, dtype=torch.int64),
        tensorize_graph_method_batch(graph_batch, max_ast_length),
    )


def test_MyDataset():
    dir = Path('/data/share/kingxu/data/CUP/clean_resub_cup2')
    type = 'c_train'
    with open('/data/share/kingxu/data/CUP/ast_type_dict.json') as f:
        type_to_id = json.load(f)
    id_to_type = {v: k for k, v in type_to_id.items()}
    data = MyDataset(dir, type, 160, 160, 230, id_to_type)
    # print(data[0])
    # print(data[1])
    # print(len(data))
    return data


def test_Dataloader():
    data = test_MyDataset()
    from dpu_utils.mlutils import Vocabulary
    from constants import (DROPOUT_RATE, EMBEDDING_PATH, FREEZE_EMBEDDING,
                           LOAD_EMBEDDINGS, NL_EMBEDDING_SIZE,
                           NODE_EMBEDDING_SIZE, SRC_EMBEDDING_SIZE,
                           VOCAB_FILE)
    sampler = SequentialSampler(data)
    with open(VOCAB_FILE, 'r') as f:
        obj = json.load(f)

    nl_vocab = Vocabulary()
    nl_vocab.token_to_id = obj['nl_vocab']['token_to_id']
    nl_vocab.id_to_token = obj['nl_vocab']['id_to_token']

    embedding_store = EmbeddingStore(nl_vocab, NL_EMBEDDING_SIZE, DROPOUT_RATE, len(SrcType),
                                     SRC_EMBEDDING_SIZE, NODE_EMBEDDING_SIZE, LOAD_EMBEDDINGS, 4,
                                     EMBEDDING_PATH, FREEZE_EMBEDDING)
    collate_fn = partial(build_batch, embedding_store=embedding_store,
                         max_nl_length=160, max_code_length=160, max_ast_length=230, max_script_length=16,
                         max_subtokens=8)
    dataloader = DataLoader(data, 192, sampler=sampler, num_workers=12, pin_memory=True,
                            collate_fn=collate_fn)
    for batch in dataloader:
        print(batch.graph_batch.graph_ids.is_pinned())


if __name__ == '__main__':
    # test_MyDataset()
    test_Dataloader()
