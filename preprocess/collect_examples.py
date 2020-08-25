import os

import argparse
import numpy as np
import pandas as pd
from p_tqdm import p_uimap

from constants import *
from utils import *

COLS = ['mrn', 'account', 'spacy_source_toks_packed', 'spacy_target_toks']


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
