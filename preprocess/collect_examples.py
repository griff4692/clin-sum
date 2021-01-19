import os

import argparse
import numpy as np
import pandas as pd
from p_tqdm import p_uimap

from constants import *
from utils import *


def collect_examples(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', str(mrn))
    df = pd.read_csv(os.path.join(mrn_dir, 'examples.csv'))
    cols = [col for col in list(df.columns) if col not in {'source_str', 'target_str'}]
    assert len(df) > 0
    return df[cols]


if __name__ == '__main__':
    _, _, mrns = get_mrn_status_df('valid_example')
    fn = os.path.join(out_dir, 'full_examples.csv')
    df = pd.concat(list(p_uimap(collect_examples, mrns, num_cpus=0.8)))
    df.to_csv(fn, index=False)

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
