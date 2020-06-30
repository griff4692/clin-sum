import os
from time import time

import pandas as pd

from constants import *


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
