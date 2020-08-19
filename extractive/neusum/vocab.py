from collections import Counter
import itertools
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.expanduser('~/clin-sum/'))

import argparse
import pandas as pd
from p_tqdm import p_uimap
import numpy as np

from preprocess.constants import *
from preprocess.section_utils import sent_toks_from_html


class Vocab:
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    def __init__(self):
        self.w2i = {}
        self.i2w = []
        self.support = []
        self.add_tokens([Vocab.PAD_TOKEN, Vocab.UNK_TOKEN])

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
        return -1

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


def get_tokens(mrn):
    """
    :param mrn:
    :return:
    """
    mrn_dir = os.path.join(out_dir, 'mrn', str(mrn))
    examples_fn = os.path.join(mrn_dir, 'examples.csv')
    examples_df = pd.read_csv(examples_fn)
    assert len(examples_df) > 0
    toks = []

    for row in examples_df.to_dict('records'):
        source_toks = sent_toks_from_html(row['spacy_source_toks'], convert_lower=True)
        target_toks = sent_toks_from_html(row['spacy_target_toks'], convert_lower=True)
        toks += (source_toks + target_toks)

    return toks


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate Vocabulary')
    parser.add_argument('--min_tf', default=5)

    args = parser.parse_args()

    splits_df = pd.read_csv(os.path.join(out_dir, 'splits.csv'))
    mrns = splits_df[splits_df['split'].isin(['validation', 'train'])]['mrn'].unique().tolist()
    tokens = p_uimap(get_tokens, mrns, num_cpus=0.25)
    tokens_flat = list(itertools.chain(*tokens))
    tok_cts = Counter(tokens_flat)
    vocab = Vocab()
    for t, v in tok_cts.items():
        if v >= args.min_tf or np.char.isnumeric(t):
            vocab.add_token(t, token_support=v)
    out_fn = os.path.join('data', 'vocab.pk')
    with open(out_fn, 'wb') as fd:
        pickle.dump(vocab, fd)
    print('Done! Saved vocabulary of size={} to {}'.format(len(vocab), out_fn))
