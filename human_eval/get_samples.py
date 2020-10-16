import os
import sys

import pandas as pd

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *


if __name__ == '__main__':
    human_dir = os.path.join(out_dir, 'human')
    if not os.path.exists(human_dir):
        os.mkdir(human_dir)

    print('Loading examples...')
    examples_df = pd.read_csv(os.path.join(out_dir, 'full_examples.csv'))
    print('Loading splits...')
    splits_df = pd.read_csv(os.path.join(out_dir, 'splits.csv'))
    hiv_mrns = set(splits_df[splits_df['hiv']]['mrn'].unique())
    print('Finding {} HIV mrns in {} examples'.format(len(hiv_mrns), len(examples_df)))

    hiv_full_df = examples_df[examples_df['mrn'].isin(hiv_mrns)]
    hiv_valid_df = hiv_full_df[hiv_full_df['split'] == 'validation']
    print('Found {} examples in full.  {} in validation'.format(len(hiv_full_df), len(hiv_valid_df)))
    # TODO fix HIV mrn detection
    # hiv_full_df = examples_df[examples_df['hiv']]
    # hiv_valid_df = hiv_full_df[hiv_full_df['split'] == 'validation']
    out_full_fn = os.path.join(human_dir, 'hiv_examples.csv')
    out_valid_fn = os.path.join(human_dir, 'hiv_validation_examples.csv')

    print('Saving {} HIV examples to {}.'.format(len(hiv_full_df), out_full_fn))
    print('Saving {} HIV examples from validation set to {}.'.format(len(hiv_valid_df), out_valid_fn))
    hiv_full_df.to_csv(out_full_fn, index=False)
    hiv_valid_df.to_csv(out_valid_fn, index=False)
