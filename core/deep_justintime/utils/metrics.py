from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(predicted_labels, true_labels):
  predicted_labels = [label.item() for label in predicted_labels]
  gold_labels = [label.item() for label in true_labels]

  assert len(predicted_labels) == len(gold_labels)

  accuracy = accuracy_score(gold_labels, predicted_labels)
  precision = precision_score(gold_labels, predicted_labels, zero_division=0)
  recall = recall_score(gold_labels, predicted_labels, zero_division=0)
  f1 = f1_score(gold_labels, predicted_labels, zero_division=0)

  return {'precision': precision, 'recall': recall, 'f1': f1, 'acc': accuracy}