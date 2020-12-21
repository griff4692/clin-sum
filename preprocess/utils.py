from datetime import datetime
import os
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from time import time

import pandas as pd

from preprocess.constants import *


def get_records(split='validation', mini=False):
    if mini:
        in_fn = os.path.join(out_dir, 'full_examples_small.csv')
    else:
        in_fn = os.path.join(out_dir, 'full_examples.csv')
    print('Loading full examples from {}'.format(in_fn))
    examples_df = pd.read_csv(in_fn)
    output_df = examples_df
    if not mini:
        if type(split) == str:
            output_df = examples_df[examples_df['split'] == split]
        elif type(split) == list:
            output_df = examples_df[examples_df['split'].isin(set(split))]
        else:
            raise Exception('Split argument must either be string or a list of strings')

    output_n = len(output_df)
    mrn_n = len(output_df['mrn'].unique())
    print('Returning {} examples for {} MRNS from {} set'.format(output_n, mrn_n, split))
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


def str_to_dt(str, trunc=False):
    dt = datetime.strptime(str, '%Y-%m-%d-%H.%M.%S.%f')
    if trunc:
        return dt.replace(minute=0, second=0, microsecond=0)
    return dt


def update_mrn_status_df(mrn_status_df, status_arr, mrn_valid_idxs, col_name):
    mrn_status_fn = os.path.join(out_dir, 'mrn_status.csv')
    n = mrn_status_df.shape[0]
    mrn_status_df[col_name] = [0] * n
    mrn_status_df[col_name].loc[mrn_valid_idxs] = status_arr
    mrn_status_df.to_csv(mrn_status_fn, index=False)
