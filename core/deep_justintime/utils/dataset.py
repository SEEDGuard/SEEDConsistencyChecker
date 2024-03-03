import torch
import pandas as pd
import os

from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import logging, LongformerTokenizer
from .constants import *


class CocoDataset(Dataset):
  def __init__(self, df):
    logging.set_verbosity_error()
    self.df = df
    self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    self.data = self.load_data(self.df)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index]

  def load_data(self, df):
      token_ids = []
      mask_ids = []
      seg_ids = []
      labels = []

      code_list = df['span_diff_code_subtokens'].to_list()
      comment_list = df['old_comment_raw'].to_list()
      label_list = df['label'].to_list()

      for (code, comment, label) in zip(code_list, comment_list, label_list):
        code_id = self.tokenizer.encode(code, add_special_tokens=False, truncation=True, max_length=MAX_LEN)
        comment_id = self.tokenizer.encode(comment, add_special_tokens=False, truncation=True, max_length=MAX_LEN)

        pair_token_ids = [self.tokenizer.cls_token_id] + comment_id + [self.tokenizer.sep_token_id] + code_id + [self.tokenizer.sep_token_id]
        pair_token_ids = self.truncate(pair_token_ids)
        code_len = len(code_id)
        comment_len = len(comment_id)

        attention_mask_ids = torch.tensor([1] * (code_len + comment_len + 3))
        segment_ids = torch.tensor([0] * (code_len + comment_len + 3))

        attention_mask_ids = self.truncate(attention_mask_ids)
        segment_ids = self.truncate(segment_ids)

        token_ids.append(torch.tensor(pair_token_ids))
        mask_ids.append(attention_mask_ids)
        seg_ids.append(segment_ids)
        labels.append(label)

      token_ids = pad_sequence(token_ids, batch_first=True)
      mask_ids = pad_sequence(mask_ids, batch_first=True)
      seg_ids = pad_sequence(seg_ids, batch_first=True)
      labels = torch.tensor(labels)

      dataset = TensorDataset(token_ids, mask_ids, seg_ids, labels)
      return dataset

  def truncate(self, ids):
    return ids[:MAX_LEN] if len(ids) > MAX_LEN else ids

# def retrieve_train_data():
#   train_param = pd.read_json(os.path.join(dataset_path, "Param", "train.json"))
#   train_return = pd.read_json(os.path.join(dataset_path, "Return", "train.json"))
#   train_summary = pd.read_json(os.path.join(dataset_path, "Summary", "train.json"))
#   train_df = pd.concat([train_param, train_return, train_summary], axis=0)
#   return train_df

# def retrieve_valid_data():
#   valid_param = pd.read_json(os.path.join(dataset_path, "Param", "valid.json"))
#   valid_return = pd.read_json(os.path.join(dataset_path, "Return", "valid.json"))
#   valid_summary = pd.read_json(os.path.join(dataset_path, "Summary", "valid.json"))
#   valid_df = pd.concat([valid_param, valid_return, valid_summary], axis=0)
#   return valid_df

def retrieve_test_data():
  test_param = pd.read_json(os.path.join(dataset_path, "Param", "test.json"))
  test_return = pd.read_json(os.path.join(dataset_path, "Return", "test.json"))
  test_summary = pd.read_json(os.path.join(dataset_path, "Summary", "test.json"))
  test_df = pd.concat([test_param, test_return, test_summary], axis=0)
  return test_df