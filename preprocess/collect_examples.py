import os

import argparse
import numpy as np
import pandas as pd
from p_tqdm import p_uimap

from constants import *
from utils import *

COLS = [
    'mrn', 'account', 'spacy_source_toks', 'spacy_target_toks', 'spacy_source_tok_ct', 'spacy_target_tok_ct',
    'coverage', 'density', 'compression', 'fragments'
]


def collect_examples(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', str(mrn))
    return pd.read_csv(os.path.join(mrn_dir, 'examples.csv'))[COLS]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate and visualize (valid) examples.')
    args = parser.parse_args()

    _, _, mrns = get_mrn_status_df('valid_example')
    n = len(mrns)

    examples = list(p_uimap(collect_examples, mrns))
    print('Concatenating all {} dataframes'.format(len(examples)))
    df = pd.concat(examples)

    out_fn = os.path.join(out_dir, 'full_examples.csv')
    print('Now saving {} examples to {}'.format(len(df), out_fn))
    df.to_csv(out_fn, index=False)

    mrns = df['mrn'].unique().tolist()
    small_mrns = set(np.random.choice(mrns, size=100, replace=False))
    tiny_mrns = set(np.random.choice(list(small_mrns), size=10, replace=False))

    small_fn = os.path.join(out_dir, 'full_examples_small.csv')
    small_df = df[df['mrn'].isin(small_mrns)]
    print('Saving {} examples to {}'.format(len(small_df), small_fn))
    small_df.to_csv(small_fn, index=False)

    tiny_fn = os.path.join(out_dir, 'full_examples_tiny.csv')
    tiny_df = small_df[small_df['mrn'].isin(tiny_mrns)]
    print('Saving {} examples to {}'.format(len(tiny_df), tiny_fn))
    tiny_df.to_csv(tiny_fn, index=False)
