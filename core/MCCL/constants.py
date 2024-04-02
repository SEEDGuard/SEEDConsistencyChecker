import os

START = '<sos>'
END = '<eos>'
NL_EMBEDDING_SIZE = 64
CODE_EMBEDDING_SIZE = 64
HIDDEN_SIZE = 64
DROPOUT_RATE = 0.6
NUM_LAYERS = 2
LR = 0.001
BATCH_SIZE = 100
MAX_EPOCHS = 100
PATIENCE = 10
VOCAB_CUTOFF_PCT = 5
LENGTH_CUTOFF_PCT = 95
MAX_VOCAB_EXTENSION = 50
BEAM_SIZE = 20
MAX_VOCAB_SIZE = 10000
FEATURE_DIMENSION = 128
NUM_CLASSES = 2

GNN_HIDDEN_SIZE = 64
GNN_LAYER_TIMESTEPS = 8
GNN_DROPOUT_RATE = 0.0
SRC_EMBEDDING_SIZE = 8
NODE_EMBEDDING_SIZE = 64

MODEL_LAMBDA = 0.5
LIKELIHOOD_LAMBDA = 0.3
OLD_METEOR_LAMBDA = 0.2
GEN_MODEL_LAMBDA = 0.5
GEN_OLD_BLEU_LAMBDA = 0.5
DECODER_HIDDEN_SIZE = 128
MULTI_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2

# Download data from here: https://drive.google.com/drive/folders/1heqEQGZHgO6gZzCjuQD1EyYertN4SAYZ?usp=sharing
# DATA_PATH should point to the location in which the above data is saved locally
DEFAULT_DATA_PATH = './core/MCCL/data/RealData/public-inconsistency-detection-data' 
#DATA_PATH = '../ExampleData/inconsistency-detection-data' 
DEFAULT_RESOURCES_PATH = os.path.join(DEFAULT_DATA_PATH, 'resources')

# Download model resources from here: https://drive.google.com/drive/folders/1cutxr4rMDkT1g2BbmCAR2wqKTxeFH11K?usp=sharing
# MODEL_RESOURCES_PATH should point to the location in which the above resources are saved locally.
MODEL_RESOURCES_PATH = './core/MCCL/data/RealData/inconsistency-detection-model-resources' # TODO
NL_EMBEDDING_PATH = os.path.join(MODEL_RESOURCES_PATH, 'nl_embeddings.json')
CODE_EMBEDDING_PATH = os.path.join(MODEL_RESOURCES_PATH, 'code_embeddings.json')
FULL_GENERATION_MODEL_PATH = os.path.join(MODEL_RESOURCES_PATH, 'generation-model.pkl.gz')

# Should point to where the output is to be saved
DETECTION_DIR = '../output' # TODO