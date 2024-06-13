import re

import javalang
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from diff_utils import is_edit_keyword


class Lemmatizer:
    def __init__(self):
        self.cache = {}
        self.tag_dict = {'J': wordnet.ADJ,
                         'N': wordnet.NOUN,
                         'V': wordnet.VERB,
                         'R': wordnet.ADV}
        self.wnl = WordNetLemmatizer()

    def lemmatize(self, word):
        if word in self.cache:
            return self.cache[word]
        else:
            lemma = self.wnl.lemmatize(word, self.get_wordnet_pos(word))
            self.cache[word] = lemma
            return lemma

    def get_wordnet_pos(self, word):
        tag = pos_tag([word])[0][1][0].upper()
        return self.tag_dict.get(tag, wordnet.NOUN)


def tokenize_subtokenize_comment(comment_raw):
    """
    输入:comment_raw
    输出:tokens, subtokens
    """
    tokens = tokenize_comment(comment_raw)  # List[str]
    all_subtokens = []
    for token in tokens:
        subtokens = subtokenize_token(token)
        all_subtokens.extend(subtokens)
    return tokens, all_subtokens


def tokenize_comment(comment_raw):
    """
    return: List[str]
    """
    comment_line = remove_html_tag(comment_raw)
    tokens = re.findall(r"[a-zA-Z0-9_]+|[^\sa-zA-Z0-9_]", comment_line.strip())
    return tokens


def subtokenize_token(token):
    """
    return: List[str] str is lowercase
    """
    if is_edit_keyword(token):
        return [token]
    token = token.replace('_', ' ')
    token = re.sub(r'([A-Z][a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', token))
    token = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', token)
    token = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', token)

    try:
        curr = [c for c in re.findall(r"[a-zA-Z0-9_]+|[^\sa-zA-Z0-9_]", token.encode('ascii', errors='ignore').decode()) if len(c) > 0]
    except:
        curr = token.split()
    subtokens = [c.lower() for c in curr]

    return subtokens


def tokenize_clean_code(code_raw):
    """
    return: List[str]
    """
    try:
        return get_clean_code(list(javalang.tokenizer.tokenize(code_raw)))
    except:
        return re.findall(r"[a-zA-Z0-9_]+|[^\sa-zA-Z0-9_]", code_raw.strip())


def get_clean_code(tokenized_code):
    """
    param: tokenized_code: output of javalang.tokenizer.tokenize()
    return: List[str]
    """
    token_vals = [t.value for t in tokenized_code]
    new_token_vals = []
    for t in token_vals:
        n = [c for c in re.findall(r"[a-zA-Z0-9_]+|[^\sa-zA-Z0-9_]", t.encode('ascii', errors='ignore').decode().strip()) if len(c) > 0]
        new_token_vals.extend(n)
    token_vals = new_token_vals
    cleaned_code_tokens = []
    for c in token_vals:
        try:
            cleaned_code_tokens.append(str(c))
        except:
            pass
    return cleaned_code_tokens


SPECIAL_TAGS = ['{', '}', '@code', '@docRoot', '@inheritDoc', '@link', '@linkplain', '@value']


def remove_html_tag(line):
    for tag in SPECIAL_TAGS:
        line = line.replace(tag, '')
    return line
