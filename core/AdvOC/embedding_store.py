import pickle

import torch
from dpu_utils.mlutils import Vocabulary
from torch import nn


class EmbeddingStore(nn.Module):
    def __init__(self, vocab, embedding_size, dropout_rate,
                 num_src_embeddings, src_embedding_size, node_embedding_size,
                 load_pretrained_embeddings=False, features_size=0,
                 embedding_path=None, freeze_embedding=False):
        """Keeps track of the NL and code vocabularies and embeddings."""
        super(EmbeddingStore, self).__init__()
        self.__nl_vocabulary = vocab
        self.__nl_embedding_layer = nn.Embedding(num_embeddings=len(vocab),
                                                 embedding_dim=embedding_size,
                                                 padding_idx=vocab.get_id_or_unk(Vocabulary.get_pad()))
        self.nl_embedding_dropout_layer = nn.Dropout(p=dropout_rate)

        self.__code_vocabulary = vocab
        self.__code_embedding_layer = self.__nl_embedding_layer
        self.code_embedding_dropout_layer = nn.Dropout(p=dropout_rate)

        self.src_embedding_layer = nn.Embedding(num_embeddings=num_src_embeddings, embedding_dim=src_embedding_size)
        self.src_embedding_dropout_layer = nn.Dropout(p=dropout_rate)
        self.node_synthesis_layer = nn.Linear(node_embedding_size*2+src_embedding_size+features_size,
                                              node_embedding_size, bias=False)

        # get_node_embeddings接收的lookup_ids多了一个维度,变成了[S, N],之前是[S,]使用RNN建模subtokens
        self.gru_layer = nn.GRU(embedding_size, node_embedding_size,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)
        self.path_gru_layer = nn.GRU(embedding_size, node_embedding_size,
                                     num_layers=1,
                                     bidirectional=True,
                                     batch_first=True)

        self.script_synthesis_layer = nn.Linear(src_embedding_size+node_embedding_size*8, node_embedding_size, bias=False)
        self.node_dropout_layer = nn.Dropout(p=dropout_rate)
        self.script_dropout_layer = nn.Dropout(p=dropout_rate)

        print('NL vocabulary size: {}'.format(len(vocab)))
        print('Code vocabulary size: {}'.format(len(vocab)))

        if load_pretrained_embeddings:
            self.initialize_embeddings(embedding_path=embedding_path, freeze_embedding=freeze_embedding)

    def initialize_embeddings(self, embedding_path, freeze_embedding=False):
        with open(embedding_path, 'rb') as f:
            embeddings = pickle.load(f)

        self.__nl_embedding_layer.weight = torch.nn.Parameter(torch.FloatTensor(embeddings),
                                                              requires_grad=not freeze_embedding)

        assert self.__nl_embedding_layer is self.__code_embedding_layer

    def get_nl_embeddings(self, token_ids):
        return self.nl_embedding_dropout_layer(self.__nl_embedding_layer(token_ids))

    def get_code_embeddings(self, token_ids):
        return self.code_embedding_dropout_layer(self.__code_embedding_layer(token_ids))

    def get_src_embeddings(self, src_ids):
        return self.src_embedding_dropout_layer(self.src_embedding_layer(src_ids))

    def get_node_embeddings(self, lookup_ids, src_ids, node_features=None):
        """
        注意返回的node_embeddings的最后一维大小是初始化EmbeddingStore的node_embedding_size
        """
        lookup_embeddings = self.get_code_embeddings(lookup_ids)
        s = lookup_embeddings.size(0)
        _, h = self.gru_layer(lookup_embeddings)
        out = h.permute(1, 0, 2).reshape([s, -1])
        src_embeddings = self.get_src_embeddings(src_ids)

        embeddings = torch.cat([out, src_embeddings], dim=-1)
        if node_features is not None:
            embeddings = torch.cat([embeddings, node_features], dim=-1)
        node_embeddings = self.node_synthesis_layer(embeddings)
        return self.node_dropout_layer(node_embeddings)

    def get_script_embeddings(self, actions, old_values, new_values, old_path, new_path):
        action_embeddings = self.get_src_embeddings(actions)
        bs, l, t = old_values.size()
        old_value_embeddings = self.get_code_embeddings(old_values)
        new_value_embeddings = self.get_code_embeddings(new_values)
        old_path_embeddings = self.get_code_embeddings(old_path)
        new_path_embeddings = self.get_code_embeddings(new_path)
        _, h1 = self.gru_layer(old_value_embeddings.view(bs * l, t, -1))
        out1 = h1.permute(1, 0, 2).reshape([bs, l, -1])
        _, h2 = self.gru_layer(new_value_embeddings.view(bs * l, t, -1))
        out2 = h2.permute(1, 0, 2).reshape([bs, l, -1])
        _, h3 = self.path_gru_layer(old_path_embeddings.view(bs * l, t, -1))
        out3 = h3.permute(1, 0, 2).reshape([bs, l, -1])
        _, h4 = self.path_gru_layer(new_path_embeddings.view(bs * l, t, -1))
        out4 = h4.permute(1, 0, 2).reshape([bs, l, -1])
        script_embeddings = self.script_synthesis_layer(torch.cat([action_embeddings, out1, out2, out3, out4], dim=-1))
        mask = actions.ne(0)  # pad位置为0 [B, L]
        mask[:, 0] = 1  # 第一个action不是pad
        return self.script_dropout_layer(torch.tanh(script_embeddings) * mask.unsqueeze(-1).float()), mask

    @property
    def nl_vocabulary(self):
        return self.__nl_vocabulary

    @property
    def code_vocabulary(self):
        return self.__code_vocabulary

    @property
    def nl_embedding_layer(self):
        return self.__nl_embedding_layer

    @property
    def code_embedding_layer(self):
        return self.__code_embedding_layer

    def get_padded_code_ids(self, code_sequence, pad_length):
        return self.__code_vocabulary.get_id_or_unk_multiple(code_sequence,
                                                             pad_to_size=pad_length,
                                                             padding_element=self.__code_vocabulary.get_id_or_unk(
                                                                 Vocabulary.get_pad()),
                                                             )

    def get_padded_nl_ids(self, nl_sequence, pad_length):
        return self.__nl_vocabulary.get_id_or_unk_multiple(nl_sequence,
                                                           pad_to_size=pad_length,
                                                           padding_element=self.__nl_vocabulary.get_id_or_unk(
                                                               Vocabulary.get_pad()),
                                                           )

    def get_extended_padded_nl_ids(self, nl_sequence, pad_length, inp_ids, inp_tokens):
        # Derived from: https://github.com/microsoft/dpu-utils/blob/master/python/dpu_utils/mlutils/vocabulary.py
        nl_ids = []
        for token in nl_sequence:
            nl_id = self.get_nl_id(token)
            if self.is_nl_unk(nl_id) and token in inp_tokens:
                copy_idx = inp_tokens.index(token)
                nl_id = inp_ids[copy_idx]
            nl_ids.append(nl_id)

        if len(nl_ids) > pad_length:
            return nl_ids[:pad_length]
        else:
            padding = [self.__nl_vocabulary.get_id_or_unk(Vocabulary.get_pad())] * (pad_length - len(nl_ids))
            return nl_ids + padding

    def pad_length(self, sequence, target_length):
        if len(sequence) >= target_length:
            return sequence[:target_length]
        else:
            return sequence + [self.__nl_vocabulary.get_id_or_unk(Vocabulary.get_pad()) for _ in range(target_length-len(sequence))]

    def get_code_id(self, token):
        return self.__code_vocabulary.get_id_or_unk(token)

    def is_code_unk(self, id):
        return id == self.__code_vocabulary.get_id_or_unk(Vocabulary.get_unk())

    def get_code_token(self, token_id):
        return self.__code_vocabulary.get_name_for_id(token_id)

    def get_nl_id(self, token):
        return self.__nl_vocabulary.get_id_or_unk(token)

    def is_nl_unk(self, id):
        return id == self.__nl_vocabulary.get_id_or_unk(Vocabulary.get_unk())

    def get_nl_token(self, token_id):
        return self.__nl_vocabulary.get_name_for_id(token_id)

    def get_vocab_extended_nl_token(self, token_id, inp_ids, inp_tokens):
        if token_id < len(self.__nl_vocabulary):
            return self.get_nl_token(token_id)
        elif token_id in inp_ids:
            copy_idx = inp_ids.index(token_id)
            return inp_tokens[copy_idx]
        else:
            return Vocabulary.get_unk()

    def get_nl_tokens(self, token_ids, inp_ids, inp_tokens, end_token):
        tokens = [self.get_vocab_extended_nl_token(t, inp_ids, inp_tokens) for t in token_ids]
        if end_token in tokens:
            return tokens[:tokens.index(end_token)]
        return tokens

    def get_end_id(self, end_token):
        return self.get_nl_id(end_token)

    def get_nl_pad_id(self):
        return self.__nl_vocabulary.get_id_or_unk(Vocabulary.get_pad())

    def get_code_pad_id(self):
        return self.__code_vocabulary.get_id_or_unk(Vocabulary.get_pad())
