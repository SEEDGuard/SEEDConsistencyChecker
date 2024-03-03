from transformers import logging, LongformerForSequenceClassification
from .constants import *

def get_model():
    logging.set_verbosity_error()
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=NUM_CLASSES)
    return model