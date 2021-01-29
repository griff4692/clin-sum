from datetime import datetime
import os
import re
import warnings
warnings.filterwarnings('error')

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from p_tqdm import p_uimap

from constants import *
from utils import *


def str_to_d(date_str):
    return datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d').date()


def str_to_dt(date_str):
    if len(date_str.split(' ')) == 1:
        return datetime.strptime(date_str, '%Y-%m-%d')
    if '.' in date_str:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')


def get_visit_id(date, visit_records):
    """
    :param date:
    :param visit_records:
    :return:
    """
    assert not date == NULL_STR
    for idx in range(len(visit_records)):
        r = visit_records[idx]
        if r['start_date'] <= date <= r['end_date']:
            return r['account']
    return NULL_STR


def is_dsum(notetype, title):
    if notetype == NULL_STR:
        if title == NULL_STR:
            return False
        tl = title.lower()
        tl_nurse = 'nurse' in tl or 'nursing' in tl
        title_dc = 'discharge' in tl and ('summary' in tl or 'note' in tl)
        if title_dc and not tl_nurse:
            return True
        return False
    ntl = notetype.lower()
    dc = 'discharge summary' in ntl or 'discharge note' in ntl
    nurse = 'nurse' in ntl or 'nursing' in ntl
    return dc and not nurse


def remove_overlapping(visit_records):
    n = len(visit_records)
    vr = []
    for i in range(len(visit_records)):
        no_prev_overlap = i == 0 or visit_records[i - 1]['end_date'] < visit_records[i]['start_date']
        no_next_overlap = i == n - 1 or visit_records[i]['end_date'] < visit_records[i + 1]['start_date']
        if no_prev_overlap and no_next_overlap:
            vr.append(visit_records[i])
    return vr


def join(mrn):
    mrn_visit_record = visit_df[visit_df['mrn'] == mrn].to_dict('records')
    num_visits = len(mrn_visit_record)
    assert num_visits > 0

    non_overlapping_visit_records = remove_overlapping(mrn_visit_record)
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    notes_fn = os.path.join(mrn_dir, 'notes.csv')
    valid_fn = os.path.join(mrn_dir, 'valid_accounts.csv')

    notes_df = pd.read_csv(notes_fn)
    notes_df.fillna(NULL_STR, inplace=True)
    note_dates = notes_df['timestamp'].apply(str_to_d).tolist()
    notes_df['account'] = list(map(lambda x: get_visit_id(x, non_overlapping_visit_records), note_dates))

    # Needs to be associated with a visit, have content, and have a timestamp
    notes_df.drop_duplicates(subset=['note_id'], inplace=True)
    notes_df['timestamp'] = notes_df['timestamp'].apply(str_to_dt)
    notes_df.sort_values('timestamp', inplace=True)
    notes_df.reset_index(inplace=True, drop=True)
    notes_n = notes_df.shape[0]

    assert notes_n > 0

    notes_df['is_target'] = notes_df['note_type'].combine(notes_df['title'], is_dsum)
    is_source = [False] * notes_n
    accounts = list(filter(lambda x: not x == NULL_STR, notes_df['account'].unique().tolist()))
    valid_accounts = []
    for account in accounts:
        assert not account in [None, NULL_STR]
        note_account_df = notes_df[notes_df['account'] == account]
        dsums = note_account_df[note_account_df['is_target']]
        n_dsums = dsums.shape[0]
        has_target = n_dsums >= 1
        has_source = False
        dsum_timestamp = dsums['timestamp'].max() if has_target else None

        if has_target:
            for x, note_row in note_account_df.iterrows():
                note_row = note_row.to_dict()
                is_pre_dsum = False if dsum_timestamp is None else note_row['timestamp'] < dsum_timestamp
                ntl = note_row['note_type'].lower()
                tl = note_row['title'].lower()
                dsum_related = 'discharge' in ntl or 'discharge' in tl
                is_source[x] = is_pre_dsum and not dsum_related
                has_source = has_source or is_source[x]
        is_valid = has_source and has_target
        if is_valid:
            valid_accounts.append(account)

    notes_df['is_source'] = is_source
    notes_df.to_csv(notes_fn, index=False)
    valid_df = pd.DataFrame(valid_accounts, columns=['account'])
    if len(valid_accounts) > 0:
        valid_df.to_csv(valid_fn, index=False)
        return 1
    return 0


if __name__ == '__main__':
    visit_df = pd.read_csv(os.path.join(out_dir, 'visits.csv'))
    print('Preprocessing visit dataframe...')
    visit_df['start_date'] = visit_df['admit_date_min'].apply(str_to_d)
    visit_df['end_date'] = visit_df['discharge_date_max'].apply(str_to_d)
    visit_df = visit_df[['account', 'mrn', 'start_date', 'end_date']]
    visit_df['mrn'] = visit_df['mrn'].astype('str')
    visit_df['account'] = visit_df['account'].astype('str')

    mrn_status_df, mrn_valid_idxs, mrns = get_mrn_status_df('note_status')
    n = len(mrns)
    print('Processing {} mrns'.format(n))

    statuses = list(p_uimap(join, mrns))
    update_mrn_status_df(mrn_status_df, statuses, mrn_valid_idxs, 'valid_account')
