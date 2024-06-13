import nltk
from nltk.corpus import stopwords

nltk.data.path.insert(0, '/data/share/kingxu/nltk_data')


stop_words = set(stopwords.words('english'))
java_keywords = set(['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class',
                     'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally',
                     'float', 'for', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long',
                     'native', 'new', 'null', 'package', 'private', 'protected', 'public', 'return', 'short',
                     'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient',
                     'try', 'void', 'volatile', 'while'])

tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
        'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB',
        'OTHER']

NUM_CODE_FEATURES = 19
NUM_NL_FEATURES = 17 + len(tags)


def get_num_code_features():
    return NUM_CODE_FEATURES


def get_num_nl_features():
    return NUM_NL_FEATURES


def is_java_keyword(token):
    return token in java_keywords


def is_operator(token):
    for s in token:
        if s.isalnum():
            return False
    return True


def get_old_code(example):
    return example.old_code_raw


def get_new_code(example):
    return example.new_code_raw


