# Your core implementation of your methods or initializing the classess
# Feel free to rename file to __init__.py if you want initialize the class for your method.
import os
import torch
import numpy as np
import random

from .utils.constants import *
from .utils.model import *
from .utils.metrics import *
from .utils.dataset import *

class Deep_JustInTime_Model:
    def __init__(self, input_path=None):
        seed = DEFAULT_SEED

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        self.classifier = get_model()

        if input_path is None:
            test_df = retrieve_test_data()
        else:
            test_df = retrieve_input_data(input_path)

        test_data = CocoDataset(test_df)
        self.test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.load_state_dict(torch.load(model_path if input_path is not None else model_path_test, map_location=self.device), strict=False)
        self.classifier.to(self.device)
    def evaluate_test_dataset(self, stats, output_path=None, num_batches=None):
        self.classifier.eval()
        test_loss = 0.0
        predictions = []
        gold_labels = []

        with torch.no_grad():
            print("Size of test dataset: ", len(self.test_loader) * BATCH_SIZE)
            for batch_idx, (sequence, attention_masks, token_type_ids, labels) in enumerate(self.test_loader):
                if num_batches is not None and num_batches == batch_idx:
                    break
                print("Batch number: ", batch_idx)
                sequence = sequence.to(self.device)
                attention_masks = attention_masks.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)

                model_output = self.classifier(sequence, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
                loss, prediction = model_output.loss, model_output.logits
                test_loss += loss.item()
                prediction = torch.argmax(prediction, dim=-1)

                predictions.extend(prediction)
                gold_labels.extend(labels)
        
        predictions = [label.item() for label in predictions]
        gold_labels = [label.item() for label in gold_labels]
    
        with open(os.path.join(output_path, "results.txt"), "w") as output_file:
            output_file.write(f"Total number of datapoints analyzed: {len(predictions)}\n\n")
            for i in range(len(predictions)):
                predicted_label = 'inconsistent' if predictions[i] == 1 else 'consistent'
                gold_label = 'inconsistent' if gold_labels[i] == 1 else 'consistent'
                conflict = '(not matching)' if predictions[i] != gold_labels[i] else '(matching)'
                output_file.write(f"{conflict} Code-comment data batch #{i}: Model found as {predicted_label}, actual is {gold_label}\n")
            if stats == True:
                test_loss /= len(self.test_loader)
                test_metrics = compute_metrics(predictions, gold_labels)
                test_acc, test_precision, test_f1, test_recall = test_metrics['acc'], test_metrics['precision'], test_metrics['f1'], test_metrics['recall']
                print(f"\nloss: {test_loss:.3f} precision: {test_precision:.3f} recall: {test_recall:.3f} f1: {test_f1:.3f} acc: {test_acc:.3f}\n")
                output_file.write(f"\nloss: {test_loss:.3f} precision: {test_precision:.3f} recall: {test_recall:.3f} f1: {test_f1:.3f} acc: {test_acc:.3f}\n")
