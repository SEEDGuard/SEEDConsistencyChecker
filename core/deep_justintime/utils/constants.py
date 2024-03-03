import os

MAX_EPOCHS = 5
TOLERANCE = 10
BATCH_SIZE = 4
LEARNING_RATE = 1E-5
NUM_CLASSES = 2
DEFAULT_SEED = 12
MAX_LEN = 1024
ACCUM_ITERS = 8
NUM_GPUS = 8

dataset_path = os.path.join(os.getcwd(), "core/deep_justintime/dataset")
model_path = os.path.join(os.getcwd(), "core/deep_justintime/model/model.weights")