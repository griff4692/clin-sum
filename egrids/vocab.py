import json
import pickle
import os
import sys

sys.path.insert(0, os.path.expanduser('~/clin-sum/'))
from preprocess.constants import *
from preprocess.section_utils import paragraph_from_html
from preprocess.tokenize_mrns import sent_segment
from preprocess.utils import *

import argparse
import pandas as pd
import numpy as np
from p_tqdm import p_uimap
from tqdm import tqdm


class Vocab:
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    NULL_TOKEN = '<null>'

    def __init__(self):
        self.w2i = {}
        self.i2w = []
        self.support = []
        self.add_tokens([Vocab.PAD_TOKEN, Vocab.UNK_TOKEN, Vocab.NULL_TOKEN])

    def pad_id(self):
        return self.get_id(Vocab.PAD_TOKEN)

    def add_tokens(self, tokens, token_support=1):
        for tidx, token in enumerate(tokens):
            self.add_token(token, token_support=token_support)

    def add_token(self, token, token_support=1):
        if token not in self.w2i or self.w2i[token] >= self.size():
            self.w2i[token] = len(self.i2w)
            self.i2w.append(token)
            self.support.append(0)
        self.support[self.get_id(token)] += token_support
        return self.w2i[token]

    def get_id(self, token):
        if token in self.w2i:
            return self.w2i[token]
        return self.w2i[Vocab.UNK_TOKEN]

    def id_count(self, id):
        return self.support[id]

    def token_count(self, token):
        return self.id_count(self.get_id(token))

    def get_ids(self, tokens):
        return list(map(self.get_id, tokens))

    def get_token(self, id):
        return self.i2w[id]

    def get_tokens(self, ids):
        return list(map(self.get_token, ids))

    def __len__(self):
        return self.size()

    def size(self):
        return len(self.i2w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate Vocabulary for E-Grid.  This is just CUI list + null')
    base_dir = os.path.join(out_dir, 'entity')
    cui_tui_group_fn = os.path.join(base_dir, 'cui_tui_group.json')
    if os.path.exists(cui_tui_group_fn):
        with open(cui_tui_group_fn, 'r') as fd:
            cui_tui_group_map = json.load(fd)
    cuis = list(set(list(cui_tui_group_map.keys())))
    vocab = Vocab()
    vocab.add_tokens(cuis, token_support=1)
    out_fn = os.path.join('data', 'vocab.pk')
    with open(out_fn, 'wb') as fd:
        pickle.dump(vocab, fd)
    print('Done! Saved vocabulary of size={} to {}'.format(len(vocab), out_fn))
