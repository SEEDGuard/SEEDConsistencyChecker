import json
import sys
from typing import NamedTuple

import torch
import torch.nn.functional as F
from dpu_utils.mlutils import Vocabulary
from torch import einsum, matmul, nn

from ast_graph_encoder import ASTGraphEncoder
from constants import (CODE_EMBEDDING_SIZE, DROPOUT_RATE, EMBEDDING_PATH,
                       FREEZE_EMBEDDING, HIDDEN_SIZE, LOAD_EMBEDDINGS,
                       MAX_CONTEXT_LENGHT, MULTI_HEADS, NC_EDGE_TYPES,
                       NL_EMBEDDING_SIZE, NODE_EMBEDDING_SIZE, NUM_LAYERS,
                       SRC_EMBEDDING_SIZE, VOCAB_FILE)
from data_utils import (DiffEdgeType, SrcType, UpdateBatchData,
                        get_num_node_features)
from embedding_store import EmbeddingStore
from encoder import Encoder
from external_cache import get_num_code_features, get_num_nl_features
from tensor_utils import compute_attention_states


class EdgesMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias_dim=None):
        super(EdgesMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // num_heads
        self.bias_dim = bias_dim
        self.register_buffer('scale', torch.rsqrt(torch.FloatTensor([self.dim_per_head])))
        if self.bias_dim is not None:
            self.bias_embs = nn.Parameter(torch.randn(self.bias_dim, self.dim_per_head))
            self.bias_scalar = nn.Parameter(torch.randn(self.dim_per_head, 1))

    def forward(self, query, key, value, key_padding_mask=None, edges=None):
        # query: B x T x D
        # key: B x S x D
        # value: B x S x D
        # key_padding_mask: BoolTensor, mask True
        # edges: LongTensor N x 2, N is number of edge
        B, T, D = query.size()
        S = key.size(1)
        query = query.view(B, T, self.num_heads, self.dim_per_head)
        key = key.view(B, S, self.num_heads, self.dim_per_head)
        value = value.view(B, S, self.num_heads, self.dim_per_head)
        alpha = einsum('bthd,bshd->bhts', query, key)
        if edges is not None:
            bias = matmul(F.one_hot(edges[:, 1], self.bias_dim).float(), self.bias_embs)
            bias = matmul(bias, self.bias_scalar).squeeze(-1)  # [N]
            bias = torch.zeros(B * T * S, device=alpha.device).index_add_(0, edges[:, 0], bias).view(B, T, S)
            summed_keys = torch.sum(key, dim=-1)
            bias = einsum('bts,bsh->bhts', bias, summed_keys)
            alpha = alpha + bias
        alpha = alpha * self.scale
        if key_padding_mask is not None:
            alpha = alpha.masked_fill(key_padding_mask, value=-1e9)
        alpha = F.softmax(alpha, dim=-1)

        output = einsum('bshd,bhts->bthd', value, alpha)
        return output.view(B, T, D), alpha


class EncoderOutputs(NamedTuple):
    """Stores tensorized batch used in edit model."""
    encoder_hidden_states: torch.Tensor
    masks: torch.Tensor
    encoder_final_state: torch.Tensor
    code_hidden_states: torch.Tensor
    code_masks: torch.Tensor
    old_nl_hidden_states: torch.Tensor
    old_nl_masks: torch.Tensor
    old_nl_final_state: torch.Tensor
    attended_old_nl_final_state: torch.Tensor


class ModuleManager(nn.Module):
    """Utility class which helps manage related attributes of the update and detection tasks."""

    def __init__(self, use_seq, use_tree, use_script):
        super(ModuleManager, self).__init__()
        # bool
        self.use_seq = use_seq
        self.use_tree = use_tree
        self.use_script = use_script

        self.num_encoders = 0
        self.num_seq_encoders = 0
        self.out_dim = 0
        self.attention_state_size = 0
        self.max_ast_length = 0
        self.max_code_length = 0
        self.max_nl_length = 0

        print('use_seq: {}'.format(self.use_seq))
        print('use tree: {}'.format(self.use_tree))
        print('use script: {}'.format(self.use_script))
        sys.stdout.flush()

    def initialize(self, *args, **kwargs):
        """Initializes model parameters from pre-defined hyperparameters"""
        with open(VOCAB_FILE, 'r') as f:
            obj = json.load(f)

        self.max_nl_length = MAX_CONTEXT_LENGHT
        self.max_code_length = obj['max_code_length']
        self.max_vocab_extension = obj['max_vocab_extension']
        self.max_ast_length = obj['max_ast_length']

        nl_vocab = Vocabulary()
        nl_vocab.token_to_id = obj['nl_vocab']['token_to_id']
        nl_vocab.id_to_token = obj['nl_vocab']['id_to_token']

        self.embedding_store = EmbeddingStore(nl_vocab, NL_EMBEDDING_SIZE, DROPOUT_RATE, len(SrcType),
                                              SRC_EMBEDDING_SIZE, NODE_EMBEDDING_SIZE, LOAD_EMBEDDINGS, get_num_node_features(),
                                              EMBEDDING_PATH, FREEZE_EMBEDDING)

        self.out_dim = 2*HIDDEN_SIZE

        # Accounting for the old NL encoder
        self.num_encoders = 1
        self.num_seq_encoders += 1
        self.attention_state_size += 2*HIDDEN_SIZE
        # for old comment
        self.nl_encoder = Encoder(NL_EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)
        self.nl_attention_transform_matrix = nn.Parameter(torch.randn(
            self.out_dim, self.out_dim, dtype=torch.float, requires_grad=True))
        self.self_attention = nn.MultiheadAttention(self.out_dim, MULTI_HEADS, DROPOUT_RATE)
        # prepare edit seq encoding
        if self.use_seq:
            self.sequence_code_encoder = Encoder(CODE_EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)
            self.num_encoders += 1
            self.num_seq_encoders += 1

            self.attention_state_size += 2*HIDDEN_SIZE
            self.sequence_attention_transform_matrix = nn.Parameter(torch.randn(
                self.out_dim, self.out_dim, dtype=torch.float, requires_grad=True))
            self.code_sequence_multihead_attention = nn.MultiheadAttention(self.out_dim, MULTI_HEADS, DROPOUT_RATE)
        # prepare edit tree encoding
        if self.use_tree:
            self.graph_code_encoder = ASTGraphEncoder(NODE_EMBEDDING_SIZE, len(DiffEdgeType))  # hidden_size=NODE_EMBEDDING_SIZE
            self.num_encoders += 1
            self.attention_state_size += 2*HIDDEN_SIZE
            self.graph_attention_transform_matrix = nn.Parameter(torch.randn(
                NODE_EMBEDDING_SIZE, self.out_dim, dtype=torch.float, requires_grad=True))
            self.graph_multihead_attention = EdgesMultiheadAttention(self.out_dim, MULTI_HEADS, NC_EDGE_TYPES)

            # self.script_attention_transform_matrix = nn.Parameter(torch.randn(
            #     NODE_EMBEDDING_SIZE, self.out_dim, dtype=torch.float, requires_grad=True))
            # self.script_multihead_attention = nn.MultiheadAttention(self.out_dim, MULTI_HEADS, DROPOUT_RATE)

        # prepare edit script encoding
        if self.use_script:
            self.script_attention = nn.Linear(NODE_EMBEDDING_SIZE, 1, bias=False)
            self.resize_with_script = nn.Linear(self.out_dim + NODE_EMBEDDING_SIZE, self.out_dim, bias=False)  # resize

        # resize add feature
        self.code_features_to_embedding = nn.Linear(CODE_EMBEDDING_SIZE + get_num_code_features(),
                                                    CODE_EMBEDDING_SIZE, bias=False)
        self.nl_features_to_embedding = nn.Linear(
            NL_EMBEDDING_SIZE + get_num_nl_features(),
            NL_EMBEDDING_SIZE, bias=False)

        # add another encoder
        self.attended_nl_encoder = Encoder(self.out_dim, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)
        self.attended_nl_encoder_output_layer = nn.Linear(self.attention_state_size, self.out_dim, bias=False)
        self.nl_attention_layer = nn.Linear(self.out_dim, 1, bias=False)

    def get_encoder_output(self, batch_data: UpdateBatchData, device):
        """Gets hidden states, final state, and a length masks corresponding to each encoder."""
        encoder_hidden_states = None
        mask = None

        # Encode old NL
        old_nl_embedded_subtokens = self.embedding_store.get_nl_embeddings(batch_data.old_nl_ids)
        old_nl_embedded_subtokens = self.nl_features_to_embedding(torch.cat(
            [old_nl_embedded_subtokens, batch_data.nl_features], dim=-1))
        old_nl_hidden_states, old_nl_final_state = self.nl_encoder.forward(old_nl_embedded_subtokens,
                                                                           batch_data.old_nl_lengths, device)
        old_nl_masks = (torch.arange(
            old_nl_hidden_states.shape[1], device=device).view(1, -1) >= batch_data.old_nl_lengths.view(-1, 1)).unsqueeze(1)
        attention_states = compute_attention_states(old_nl_hidden_states, old_nl_masks,
                                                    old_nl_hidden_states, transformation_matrix=self.nl_attention_transform_matrix, multihead_attention=self.self_attention)

        # Encode code
        code_hidden_states = None
        code_masks = None

        if self.use_seq:
            code_embedded_subtokens = self.embedding_store.get_code_embeddings(batch_data.code_ids)
            code_embedded_subtokens = self.code_features_to_embedding(torch.cat(
                [code_embedded_subtokens, batch_data.code_features], dim=-1))
            code_hidden_states, code_final_state = self.sequence_code_encoder.forward(code_embedded_subtokens,
                                                                                      batch_data.code_lengths, device)
            code_masks = (torch.arange(
                code_hidden_states.shape[1], device=device).view(1, -1) >= batch_data.code_lengths.view(-1, 1)).unsqueeze(1)
            encoder_hidden_states = code_hidden_states

            attention_states = torch.cat([attention_states, compute_attention_states(
                code_hidden_states, code_masks, old_nl_hidden_states,
                transformation_matrix=self.sequence_attention_transform_matrix,
                multihead_attention=self.code_sequence_multihead_attention)], dim=-1)

        if self.use_tree:
            embedded_nodes = self.embedding_store.get_node_embeddings(
                batch_data.graph_batch.value_lookup_ids, batch_data.graph_batch.src_type_ids, batch_data.graph_batch.node_features)

            graph_states = self.graph_code_encoder.forward(embedded_nodes, batch_data.graph_batch, device)
            graph_lengths = batch_data.graph_batch.num_nodes_per_graph
            graph_masks = (torch.arange(
                graph_states.shape[1], device=device).view(1, -1) >= graph_lengths.view(-1, 1)).unsqueeze(1).unsqueeze(1)

            transformed_graph_states = torch.einsum('ijk,km->ijm', graph_states, self.graph_attention_transform_matrix)
            graph_attention_states, _ = self.graph_multihead_attention(old_nl_hidden_states, transformed_graph_states, transformed_graph_states,
                                                                       graph_masks, batch_data.nl_code_edges)
            attention_states = torch.cat([attention_states, graph_attention_states], dim=-1)

        
        nl_attended_states = torch.tanh(self.attended_nl_encoder_output_layer(attention_states))
        attended_old_nl_states, _ = self.attended_nl_encoder.forward(nl_attended_states,
                                                                     batch_data.old_nl_lengths, device)
        tmp = torch.arange(old_nl_hidden_states.shape[1], device=device)
        nl_masks1 = (tmp.view(1, -1) >= batch_data.old_nl_start.view(-1, 1))
        nl_masks2 = (tmp.view(1, -1) < batch_data.old_nl_end.view(-1, 1))
        nl_masks = nl_masks1 ^ nl_masks2
        self_energe = self.nl_attention_layer(attended_old_nl_states).squeeze(-1).masked_fill(nl_masks, -1e9)
        self_attention = torch.softmax(self_energe, dim=-1)
        attended_old_nl_final_state = torch.einsum('ijk,ij->ik', attended_old_nl_states, self_attention)
        if self.use_script:
            embedded_script, s_mask = self.embedding_store.get_script_embeddings(batch_data.action_ids, batch_data.old_value_ids, batch_data.new_value_ids,
                                                                                 batch_data.old_path_ids, batch_data.new_path_ids)
            script_engery = self.script_attention(embedded_script).squeeze(-1).masked_fill(~s_mask, -1e9)
            script_attention = torch.softmax(script_engery, dim=-1)
            script_state = torch.einsum('ijk,ij->ik', embedded_script, script_attention)  # NODE_EMBEDDING_SIZE

            # script_states = compute_attention_states(embedded_script, ~s_mask.unsqueeze(1), old_nl_hidden_states,
            #                                          transformation_matrix=self.script_attention_transform_matrix,
            #                                          multihead_attention=self.script_multihead_attention)

            attended_old_nl_final_state = torch.tanh(self.resize_with_script(torch.cat([attended_old_nl_final_state, script_state], dim=-1)))
            # attended_old_nl_final_state = self.resize_with_script(torch.cat([attended_old_nl_final_state, script_state], dim=-1))

        return EncoderOutputs(encoder_hidden_states, mask, None, code_hidden_states, code_masks,
                              old_nl_hidden_states, old_nl_masks, old_nl_final_state, attended_old_nl_final_state)


def test():
    query = torch.randn(3, 10, 64)
    key = torch.randn(3, 20, 64)
    mask = torch.ones(3, 20).bool()
    value = key
    edges = torch.tensor([[6, 0], [10, 0], [100, 0]], dtype=torch.long)
    layer = EdgesMultiheadAttention(64, 4, 2)
    output, alpha = layer(query, key, value, key_padding_mask=mask, edges=edges)
    print(output)
    print(alpha)


if __name__ == '__main__':
    test()
