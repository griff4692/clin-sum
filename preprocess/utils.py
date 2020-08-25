import os
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from time import time

import pandas as pd

from preprocess.constants import *


MINI_CT = 100


def get_records(type='validation', mini=False):
    if mini:
        in_fn = os.path.join(out_dir, 'full_examples_small.csv')
        if not os.path.exists(in_fn):
            examples_df = pd.read_csv(os.path.join(out_dir, 'full_examples.csv'))
            examples_df = examples_df.sample(n=15000, replace=False)
            examples_df.to_csv(in_fn, index=False)
    else:
        in_fn = os.path.join(out_dir, 'full_examples.csv')
    print('Loading full examples from {}'.format(in_fn))
    examples_df = pd.read_csv(in_fn)

    splits_fn = os.path.join(out_dir, 'splits.csv')
    splits_df = pd.read_csv(splits_fn)[['mrn', 'account', 'split']]
    subset_df = splits_df[splits_df['split'] == type]
    subset_n = len(subset_df)
    output_df = examples_df.merge(subset_df, on=['mrn', 'account'])
    output_n = len(output_df)
    mrn_n = len(output_df['mrn'].unique())

    if not mini:
        assert subset_n == output_n
    print('Returning {} examples for {} MRNS from {} set'.format(output_n, mrn_n, type))
    if mini and output_n > MINI_CT:
        print('Sampling {} for mini {} dataset'.format(MINI_CT, type))
        output_df = output_df.sample(n=MINI_CT, replace=False)
    return output_df


def duration(start_time):
    end_time = time()
    minutes = (end_time - start_time) / 60.0
    round_factor = 0
    if minutes < 1:
        round_factor = 2
    print('Took {} minutes'.format(minutes, round(round_factor)))


def get_mrn_status_df(status_col):
    mrn_status_fn = os.path.join(out_dir, 'mrn_status.csv')
    mrn_status_df = pd.read_csv(mrn_status_fn)
    mrn_status_df['mrn'] = mrn_status_df['mrn'].astype('str')
    mrn_status_df[status_col] = mrn_status_df[status_col].astype('int')
    mrn_valid_idxs = mrn_status_df[status_col] == 1
    mrns = mrn_status_df.loc[mrn_valid_idxs]['mrn'].tolist()
    return mrn_status_df, mrn_valid_idxs, mrns


def update_mrn_status_df(mrn_status_df, status_arr, mrn_valid_idxs, col_name):
    mrn_status_fn = os.path.join(out_dir, 'mrn_status.csv')
    n = mrn_status_df.shape[0]
    mrn_status_df[col_name] = [0] * n
    mrn_status_df[col_name].loc[mrn_valid_idxs] = status_arr
    mrn_status_df.to_csv(mrn_status_fn, index=False)
