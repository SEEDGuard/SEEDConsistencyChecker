
import json
import sys
from datetime import datetime
from functools import partial
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim, sigmoid
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from clf_metrics import ClfMetric
from constants import (ALPHA, AST_TYPE_DICT, BATCH_SIZE, BETA, DATA_PATH,
                       EVAL_EVERY_N_BATCHES, LOG_EVERY_N_BATCHES, LR,
                       MAX_EPOCHS, MAX_SCRIPT_LENGTH, MAX_SUBTOKENS, PATIENCE)
from data_loader import MyDataset, UpdateBatchData, build_batch
from module_manager import ModuleManager


class DetectionModule(nn.Module):
    def __init__(self, manager: ModuleManager) -> None:
        super(DetectionModule, self).__init__()
        self.manager = manager
        self.output_layer = nn.Linear(self.manager.out_dim, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, batch_data):
        """Computes prediction loss for given batch."""
        encoder_outputs = self.manager.get_encoder_output(batch_data, self.get_device())
        predictions = self.output_layer(encoder_outputs.attended_old_nl_final_state)
        return predictions

    def get_device(self):
        return self.output_layer.weight.device


def compute_score(predicted_labels, gold_labels):
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    threshold = 0.5
    assert(len(predicted_labels) == len(gold_labels))
    all = len(predicted_labels)
    for i in range(len(gold_labels)):
        if gold_labels[i]:
            if predicted_labels[i] > threshold:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted_labels[i] > threshold:
                false_positives += 1
            else:
                true_negatives += 1
    try:
        precision = true_positives/(true_positives + false_positives)
    except:
        precision = 0.0
    try:
        recall = true_positives/(true_positives + false_negatives)
    except:
        recall = 0.0
    try:
        f1 = 2*((precision * recall)/(precision + recall))
    except:
        f1 = 0.0
    try:
        score = recall * recall * all / (true_positives + false_positives)
    except:
        score = 0.0
    return precision, recall, f1, score


class PredictiveAdversaryNetworks():
    """
    Predictive Adversary Networks (PAN)
    """

    def __init__(self, model_path, d_manager, c_manager):
        super(PredictiveAdversaryNetworks, self).__init__()

        self.model_path = Path(model_path)

        self.discriminator = DetectionModule(d_manager)
        self.classifier = DetectionModule(c_manager)

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=LR)
        self.optimizerC = optim.Adam(self.classifier.parameters(), lr=LR)

    def load_model(self, model_path):
        model_path = Path(model_path)
        self.discriminator.load_state_dict(torch.load(model_path / 'discriminator.pt'))
        self.classifier.load_state_dict(torch.load(model_path / 'classifier.pt'))

    def load_discriminator(self, file):
        print(f'Loading discriminator from {file}', flush=True)
        self.discriminator.load_state_dict(torch.load(file), strict=False)

    def load_classifier(self, file):
        print(f'Loading classifier from {file}', flush=True)
        self.classifier.load_state_dict(torch.load(file), strict=False)

    def run_train(self):
        embedding_store = self.classifier.manager.embedding_store
        batch_size = BATCH_SIZE
        alpha = ALPHA
        beta = BETA
        max_nl_length = self.classifier.manager.max_nl_length
        max_code_length = self.classifier.manager.max_code_length
        max_ast_length = self.classifier.manager.max_ast_length
        device = self.classifier.get_device()

        best_score = 0.0
        patience_tally = 0

        batch_count = 0
        train_loss_d = 0.
        all_train_loss_d = 0.
        train_loss_c = 0.
        all_train_loss_c = 0.
        stop_train = False

        with open(AST_TYPE_DICT) as f:
            type_to_id = json.load(f)
        id_to_type = {v: k for k, v in type_to_id.items()}
        collate_fn = partial(build_batch, embedding_store=embedding_store,
                             max_nl_length=max_nl_length, max_code_length=max_code_length, max_ast_length=max_ast_length,
                             max_script_length=MAX_SCRIPT_LENGTH, max_subtokens=MAX_SUBTOKENS)
        data_path = Path(DATA_PATH)
        # by time
        p_dataset = MyDataset(data_path, 'c_train', max_nl_length, max_code_length, max_ast_length, id_to_type)
        u_dataset = MyDataset(data_path, 'u_train', max_nl_length, max_code_length, max_ast_length, id_to_type)

        # by project
        # p_dataset = MyDataset(data_path, 'train', max_nl_length, max_code_length, max_ast_length, id_to_type, subset='r')
        # u_dataset = MyDataset(data_path, 'train', max_nl_length, max_code_length, max_ast_length, id_to_type, subset='u')

        gamma = len(p_dataset) / len(u_dataset)

        v_dataset = MyDataset(data_path, 'valid', max_nl_length, max_code_length, max_ast_length, id_to_type)

        p_sampler = RandomSampler(p_dataset)
        p_data = cycle(DataLoader(p_dataset, batch_size, sampler=p_sampler,
                                  num_workers=3, collate_fn=collate_fn, pin_memory=True))

        for epoch in range(MAX_EPOCHS):
            self.discriminator.train()
            self.classifier.train()

            u_sampler = SequentialSampler(u_dataset)
            u_data = DataLoader(u_dataset, batch_size, sampler=u_sampler,
                                num_workers=3, collate_fn=collate_fn, pin_memory=True)
            for u_batch in u_data:
                u_batch = u_batch.to_device_new_obj(device)
                p_batch = next(p_data)
                p_batch = p_batch.to_device_new_obj(device)

                self.discriminator.zero_grad()
                output = self.discriminator(p_batch).view(-1)
                errD_p = gamma * self.criterion(output, p_batch.labels.float())
                errD_p.backward()
                output_u = self.discriminator(u_batch).view(-1)
                errD_u = alpha * self.criterion(output_u, u_batch.labels.float())
                output_c = self.classifier(u_batch).view(-1)
                labels_c = sigmoid(output_c).detach()
                errD_c = beta * self.criterion(output_u, (labels_c < 0.5).float())
                loss = errD_c + errD_u
                loss.backward()
                self.optimizerD.step()
                errD = loss.item() + errD_p.item()
                self.classifier.zero_grad()
                labels_d = sigmoid(output_u).detach()
                # errC = self.criterion(output_c, (labels_d > 0.5).float())
                errC = self.criterion(output_c, labels_d)
                errC.backward()
                self.optimizerC.step()

                batch_count += 1
                train_loss_d += errD
                all_train_loss_d += errD
                train_loss_c += errC.item()
                all_train_loss_c += errC.item()

                if batch_count % LOG_EVERY_N_BATCHES == 0:
                    print('Epoch %d, batch %d, D_loss: %.5f, C_loss: %.5f' % (epoch, batch_count, train_loss_d / LOG_EVERY_N_BATCHES, train_loss_c / LOG_EVERY_N_BATCHES))
                    sys.stdout.flush()
                    train_loss_d = 0.
                    train_loss_c = 0.
                if batch_count % EVAL_EVERY_N_BATCHES == 0 and epoch > 0:
                    print('Epoch %d, batch %d, D_loss: %.5f, C_loss: %.5f' % (epoch, batch_count, all_train_loss_d / EVAL_EVERY_N_BATCHES, all_train_loss_c / EVAL_EVERY_N_BATCHES))
                    sys.stdout.flush()
                    all_train_loss_d = 0.
                    all_train_loss_c = 0.
                    validation_loss = 0.
                    validation_predicted_labels = []
                    validation_gold_labels = []
                    valid_batch_count = 0
                    self.classifier.eval()
                    with torch.no_grad():
                        v_sampler = SequentialSampler(v_dataset)
                        v_data = DataLoader(v_dataset, batch_size, sampler=v_sampler,
                                            num_workers=3, collate_fn=collate_fn, pin_memory=True)
                        for batch in v_data:
                            batch = batch.to_device_new_obj(device)
                            valid_batch_count += 1
                            output = self.classifier(batch).view(-1)
                            valid_loss = self.criterion(output, batch.labels.float())
                            validation_loss += valid_loss.item()
                            validation_predicted_labels.extend(sigmoid(output).tolist())
                            validation_gold_labels.extend(batch.labels.tolist())
                            # if debug_flag:
                            #     break
                    p, r, f1, score = compute_score(validation_predicted_labels, validation_gold_labels)
                    validation_loss = validation_loss/valid_batch_count
                    if score >= best_score:
                        best_score = score
                        torch.save(self.classifier.state_dict(), str(self.model_path / 'classifier.pt'))
                        torch.save(self.discriminator.state_dict(), str(self.model_path / 'discriminator.pt'))
                        saved = True
                        patience_tally = 0
                    else:
                        saved = False
                        patience_tally += 1
                    print('Validation loss: {:.5f}'.format(validation_loss))
                    print('Validation precision: {:.5f}'.format(p))
                    print('Validation recall: {:.5f}'.format(r))
                    print('Validation f1: {:.5f}'.format(f1))
                    print('Validation score: {:.5f}'.format(score))
                    if saved:
                        print('Saved')
                    print('-----------------------------------')
                    sys.stdout.flush()
                    self.classifier.train()
                    if patience_tally >= PATIENCE:
                        print('Early stopping')
                        torch.save(self.classifier.state_dict(), str(self.model_path / 'classifier_last.pt'))
                        torch.save(self.discriminator.state_dict(), str(self.model_path / 'discriminator_last.pt'))
                        stop_train = True
                        break
            if stop_train:
                break

    def run_evaluation(self, model_path):
        self.discriminator = None
        self.classifier.eval()
        embedding_store = self.classifier.manager.embedding_store
        batch_size = BATCH_SIZE * 3
        max_nl_length = self.classifier.manager.max_nl_length
        max_code_length = self.classifier.manager.max_code_length
        max_ast_length = self.classifier.manager.max_ast_length
        device = self.classifier.get_device()

        results = []
        gold_labels = []

        with open(AST_TYPE_DICT) as f:
            type_to_id = json.load(f)
        id_to_type = {v: k for k, v in type_to_id.items()}
        collate_fn = partial(build_batch, embedding_store=embedding_store,
                             max_nl_length=max_nl_length, max_code_length=max_code_length, max_ast_length=max_ast_length,
                             max_script_length=MAX_SCRIPT_LENGTH, max_subtokens=MAX_SUBTOKENS)
        p_dataset = MyDataset(Path(DATA_PATH), 'verified_test', max_nl_length, max_code_length, max_ast_length, id_to_type)
        p_sampler = SequentialSampler(p_dataset)
        test_batches = DataLoader(p_dataset, batch_size, sampler=p_sampler,
                                  num_workers=6, collate_fn=collate_fn, pin_memory=True)

        with torch.no_grad():
            for batch in tqdm(test_batches):
                batch = batch.to_device_new_obj(device)
                output = self.classifier(batch).view(-1)
                results.extend(sigmoid(output).tolist())
                gold_labels.extend(batch.labels.tolist())
                # if debug_flag:
                #     break
        probs = np.array(results)
        n_probs = 1 - probs
        probs = np.vstack((n_probs, probs)).transpose()
        out = Path(model_path)
        torch.save(torch.from_numpy(probs), str(out / 'result.bin'))

        metric = ClfMetric()
        result = metric.eval(probs, np.array(gold_labels))
        print(result)
        with open(out / 'eval.log', 'a') as f:
            f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S:"))
            f.write(json.dumps(result) + '\n')


        results = []
        gold_labels = []
        p_dataset = MyDataset(Path(DATA_PATH), 'test', max_nl_length, max_code_length, max_ast_length, id_to_type)
        p_sampler = SequentialSampler(p_dataset)
        test_batches = DataLoader(p_dataset, batch_size, sampler=p_sampler,
                                  num_workers=6, collate_fn=collate_fn, pin_memory=True)

        with torch.no_grad():
            for batch in tqdm(test_batches):
                batch = batch.to_device_new_obj(device)
                output = self.classifier(batch).view(-1)
                results.extend(sigmoid(output).tolist())
                gold_labels.extend(batch.labels.tolist())
                # if debug_flag:
                #     break
        probs = np.array(results)
        n_probs = 1 - probs
        probs = np.vstack((n_probs, probs)).transpose()
        out = Path(model_path)
        torch.save(torch.from_numpy(probs), str(out / 'result.bin'))

        metric = ClfMetric()
        result = metric.eval(probs, np.array(gold_labels))
        print(result)
        with open(out / 'eval.log', 'a') as f:
            f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S:"))
            f.write(json.dumps(result) + '\n')

    def pretrain_discriminator(self):
        self.classifier = None
        embedding_store = self.discriminator.manager.embedding_store
        batch_size = BATCH_SIZE * 2
        max_nl_length = self.discriminator.manager.max_nl_length
        max_code_length = self.discriminator.manager.max_code_length
        max_ast_length = self.discriminator.manager.max_ast_length
        device = self.discriminator.get_device()
        best_score = 0.0
        patience_tally = 0
        batch_count = 0
        train_loss = 0.
        all_train_loss = 0.

        with open(AST_TYPE_DICT) as f:
            type_to_id = json.load(f)
        id_to_type = {v: k for k, v in type_to_id.items()}
        collate_fn = partial(build_batch, embedding_store=embedding_store,
                             max_nl_length=max_nl_length, max_code_length=max_code_length, max_ast_length=max_ast_length,
                             max_script_length=MAX_SCRIPT_LENGTH, max_subtokens=MAX_SUBTOKENS)
        data_path = Path(DATA_PATH)
        p_dataset = MyDataset(data_path, 'c_train', max_nl_length, max_code_length, max_ast_length, id_to_type)
        u_dataset = MyDataset(data_path, 'u_train', max_nl_length, max_code_length, max_ast_length, id_to_type)
        u_dataset.append(p_dataset)

        v_dataset = MyDataset(data_path, 'valid', max_nl_length, max_code_length, max_ast_length, id_to_type)
        for epoch in range(MAX_EPOCHS):
            self.discriminator.train()
            sampler = RandomSampler(u_dataset)
            data = DataLoader(u_dataset, batch_size, sampler=sampler, collate_fn=collate_fn, pin_memory=True, num_workers=4)
            for t_batch in data:
                t_batch = t_batch.to_device_new_obj(device)
                self.discriminator.zero_grad()
                output = self.discriminator(t_batch).view(-1)
                loss = self.criterion(output, t_batch.labels.float())
                loss.backward()
                self.optimizerD.step()

                batch_count += 1
                train_loss += loss.item()
                all_train_loss += loss.item()

                if batch_count % LOG_EVERY_N_BATCHES == 0:
                    print('Epoch %d, batch %d, D_loss: %.5f' % (epoch, batch_count, train_loss / LOG_EVERY_N_BATCHES))
                    sys.stdout.flush()
                    train_loss = 0.
                if batch_count % EVAL_EVERY_N_BATCHES == 0:
                    print('Epoch %d: Batch %d: avg train_loss: %.6f' % (epoch, batch_count, all_train_loss/EVAL_EVERY_N_BATCHES))
                    sys.stdout.flush()
                    all_train_loss = 0.
                    self.discriminator.eval()
                    validation_loss = 0
                    validation_predicted_labels = []
                    validation_gold_labels = []
                    valid_batch_count = 0
                    with torch.no_grad():
                        v_sampler = SequentialSampler(v_dataset)
                        v_data = DataLoader(v_dataset, batch_size, sampler=v_sampler,
                                            num_workers=4, collate_fn=collate_fn, pin_memory=True)
                        for batch in v_data:
                            batch = batch.to_device_new_obj(device)
                            valid_batch_count += 1
                            output = self.discriminator(batch).view(-1)
                            valid_loss = self.criterion(output, batch.labels.float())
                            validation_loss += valid_loss.item()
                            validation_predicted_labels.extend(sigmoid(output).tolist())
                            validation_gold_labels.extend(batch.labels.tolist())
                    validation_loss = validation_loss/valid_batch_count
                    p, r, f1, score = compute_score(validation_predicted_labels, validation_gold_labels)
                    if score >= best_score:
                        best_score = score
                        torch.save(self.discriminator.state_dict(), str(self.model_path / 'discriminator.pt'))
                        saved = True
                        patience_tally = 0
                    else:
                        saved = False
                        patience_tally += 1
                    print('Validation loss: {:.5f}'.format(validation_loss))
                    print('Validation precision: {:.5f}'.format(p))
                    print('Validation recall: {:.5f}'.format(r))
                    print('Validation f1: {:.5f}'.format(f1))
                    print('Validation score: {:.5f}'.format(score))
                    if saved:
                        print('Saved')
                    print('-----------------------------------')
                    sys.stdout.flush()
                    self.discriminator.train()
                    if patience_tally >= PATIENCE:
                        print('Terminating: {}'.format(epoch))
                        stop_train = True
                        break
            # print(f'Epoch {epoch}, loss: {all_train_loss / batch_count}', flush=True)
            # torch.save(self.discriminator.state_dict(), str(self.model_path / f'discriminator_{epoch}_epoch.pt'))

    # def data_to_device(self, batch_data: UpdateBatchData, device):
    #     """Moves data to device."""
    #     graph_batch = batch_data.graph_batch
    #     new_graph_batch = GraphMethodBatch(
    #         graph_batch.graph_ids.to(device=device, dtype=torch.int64),
    #         graph_batch.value_lookup_ids.to(device=device, dtype=torch.int64),
    #         graph_batch.src_type_ids.to(device=device, dtype=torch.int64),
    #         graph_batch.root_ids.to(device=device, dtype=torch.int64),
    #         graph_batch.is_internal.to(device),
    #         graph_batch.edges,
    #         graph_batch.num_graphs,
    #         graph_batch.num_nodes,
    #         graph_batch.node_features.to(device),
    #         graph_batch.node_positions.to(device=device, dtype=torch.int64),
    #         graph_batch.num_nodes_per_graph.to(device=device, dtype=torch.int64),
    #     )
    #     return UpdateBatchData(
    #         batch_data.code_ids.to(device=device, dtype=torch.int64),
    #         batch_data.code_lengths.to(device=device, dtype=torch.int64),
    #         batch_data.old_nl_ids.to(device=device, dtype=torch.int64),
    #         batch_data.old_nl_lengths.to(device=device, dtype=torch.int64),
    #         batch_data.old_nl_start.to(device=device, dtype=torch.int64),
    #         batch_data.old_nl_end.to(device=device, dtype=torch.int64),
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         batch_data.code_features.to(device),
    #         batch_data.nl_features.to(device),
    #         batch_data.labels.to(device),
    #         batch_data.action_ids.to(device=device, dtype=torch.int64),
    #         batch_data.old_value_ids.to(device=device, dtype=torch.int64),
    #         batch_data.new_value_ids.to(device=device, dtype=torch.int64),
    #         batch_data.old_path_ids.to(device=device, dtype=torch.int64),
    #         batch_data.new_path_ids.to(device=device, dtype=torch.int64),
    #         new_graph_batch,
    #         batch_data.nl_code_edges,
    #     )


# def debug_bceloss():
#     from zqkx_module_manager import ModuleManager
#     manager1 = ModuleManager(True, True, True, True, 'detect')
#     manager2 = ModuleManager(True, True, True, True, 'detect')
#     manager1.initialize()
#     manager2.initialize()
#     model = PredictiveAdversaryNetworks('', manager1, manager2)
#     max_nl_length = model.classifier.manager.max_nl_length
#     max_code_length = model.classifier.manager.max_code_length
#     max_ast_length = model.classifier.manager.max_ast_length
#     model.classifier.cuda()
#     model.discriminator.cuda()
#     device = model.classifier.get_device()
#     embedding_store = model.classifier.manager.embedding_store
#     batch_size = BATCH_SIZE
#     p_fs = get_opened_files(DATA_PATH, 'p_train')
#     i = 0
#     while i < batch_size * 200:
#         [f.readline() for f in p_fs]
#         i += 1
#     p_examples = example_generator(p_fs, max_nl_length, max_code_length, max_ast_length)
#     p_data = BackgroundGenerator(
#         batch_generator(p_examples, batch_size, embedding_store,
#                         max_nl_length, max_code_length, max_ast_length,
#                         device),
#         2)
#     for p_batch in p_data:
#         output = model.discriminator(p_batch).view(-1)
#         errD_p = model.criterion(output, p_batch.labels.float())
#         print(errD_p)


if __name__ == "__main__":
    # debug_bceloss()
    with open('/data/share/kingxu/data/CUP/clean_resub_cup2/test_seq.jsonl') as f:
        labels = []
        for line in f:
            labels.append(json.loads(line)['label'])
    probs = torch.load('/data/share/kingxu/code/deep-jit-inconsistency-detection/clean_resub_cup2_pan_seq_tree_features_random_sript/result.bin')
    metric = ClfMetric()
    result = metric.eval(probs.numpy(), np.array(labels))
    print(result)
    with open('/data/share/kingxu/code/deep-jit-inconsistency-detection/clean_resub_cup2_pan_seq_tree_features_random_sript/eval.log', 'a') as f:
        f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S:"))
        f.write(json.dumps(result) + '\n')
