from pathlib import Path

START = '<sos>'
END = '<eos>'
NL_EMBEDDING_SIZE = 64
CODE_EMBEDDING_SIZE = 64
HIDDEN_SIZE = 64
DROPOUT_RATE = 0.5
NUM_LAYERS = 2
LR = 0.0001
BATCH_SIZE = 128
ALPHA = 0.4
BETA = 0.1
MAX_EPOCHS = 100
PATIENCE = 10

LOG_EVERY_N_BATCHES = 200
EVAL_EVERY_N_BATCHES = 4000

GNN_HIDDEN_SIZE = 64  # not used
GNN_LAYER_TIMESTEPS = 8
GNN_DROPOUT_RATE = 0.0
SRC_EMBEDDING_SIZE = 8
NODE_EMBEDDING_SIZE = 64

MULTI_HEADS = 4

# by time
DATA_PATH = './'  # TODO

# by project
# DATA_PATH = './'

AST_TYPE_DICT = 'data_process/ast_type_dict.json'


graph_flag = True
if graph_flag:
    VOCAB_FILE = str(Path(DATA_PATH) / 'vocab.json')
    LOAD_EMBEDDINGS = False
    EMBEDDING_PATH = str(Path(DATA_PATH) / 'embed.pkl')
    FREEZE_EMBEDDING = False

    NODE_EMBEDDING_SIZE = 128
    MAX_SUBTOKENS = 8
    NC_EDGE_TYPES = 3  # nl-code edge
    MAX_CONTEXT_LENGHT = 160  # for module_manager
    MAX_SCRIPT_LENGTH = 10

debug_flag = False
if debug_flag:
    LOG_EVERY_N_BATCHES = 1
    EVAL_EVERY_N_BATCHES = 10
