from collections import Counter
import itertools
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.expanduser('~/clin-sum/'))

import argparse
import pandas as pd
import numpy as np
from p_tqdm import p_uimap
from tqdm import tqdm

from preprocess.constants import *
from preprocess.utils import get_records
from preprocess.section_utils import sent_toks_from_html


def rmatch(char, double=False, escape_char=True):
    esc_char = '\\' + char if escape_char else char
    if double:
        return r'\d+{}\d+{}\d+'.format(esc_char, esc_char)
    return r'\d+{}\d+'.format(esc_char)


MATCHES = [
    ('<date>', r'\d{2}\-[a-z]{2,10}\-\d{2,4}'),
    ('<age>', r'\d+\-?(yr|year)s?(old)?'),
    ('<int>', r'\d+'),
    ('<neg>', r'-[\d\W]+'),
    ('<time>', r'[\d:.]+(am|pm)'),
    ('<dot>', rmatch('.', double=False)),
    ('<ddot>', rmatch('.', double=True)),
    ('<dash>', rmatch('-', double=False)),
    ('<ddash>', rmatch('-', double=True)),
    ('<colon>', rmatch(':', double=False)),
    ('<dcolon>', rmatch(':', double=True)),
    ('<slash>', rmatch('/', double=False)),
    ('<dslash>', rmatch('/', double=True)),
    ('<xmatch>', rmatch('x', double=False, escape_char=False)),
    ('<weight>', r'\d+(lbs|lb|kg|kgs|gram|grams)'),
    ('<mg>', r'\d+mgs?'),
    ('<hours>', r'\d+hours?'),
    ('<minutes>', r'\d+minutes?'),
    ('<days>', r'\d+days?'),
    (r'<cr>', r'[\d\W]+cr')
]


def cast_num(str):
    for v, reg in MATCHES:
        if re.search('^(' + reg + ')$', str) is not None:
            return v
    return str


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
        token = cast_num(token)
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


def get_tokens(row):
    source_toks = sent_toks_from_html(row['spacy_source_toks'], convert_lower=True)
    target_toks = sent_toks_from_html(row['spacy_target_toks'], convert_lower=True)
    return source_toks + target_toks


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate Vocabulary')
    parser.add_argument('--min_tf', default=10, type=int)
    parser.add_argument('-template_numbers', default=False, action='store_true')

    args = parser.parse_args()

    if args.template_numbers:
        in_fn = os.path.join('data', 'vocab.pk')
        with open(in_fn, 'rb') as fd:
            prev_vocab = pickle.load(fd)

        vocab = Vocab()
        vocab.add_tokens([x[0] for x in MATCHES])

        for i in tqdm(range(len(prev_vocab))):
            tok = prev_vocab.i2w[i]
            sup = prev_vocab.support[i]
            if sup >= args.min_tf:
                tok_adj = cast_num(tok)
                vocab.add_token(tok_adj, sup)

        out_fn = os.path.join('data', 'vocab_num_template.pk')
        print('Vocab reduced from {} to {}'.format(len(prev_vocab), len(vocab)))
        print('Saving it now to {}'.format(out_fn))
        with open(out_fn, 'wb') as fd:
            pickle.dump(vocab, fd)
    else:
        records = get_records(split=['train', 'validation']).to_dict('records')
        tokens = p_uimap(get_tokens, records, num_cpus=0.8)
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
