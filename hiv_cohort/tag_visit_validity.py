from datetime import datetime
import os
from multiprocessing import Pool
import shutil
from time import time

import pandas as pd
from tqdm import tqdm

notes_dir = '/nlp/projects/clinsum/notes_by_mrn'


def str_to_dt(date_str):
    if len(date_str.split(' ')) == 1:
        return datetime.strptime(date_str, '%Y-%m-%d')
    if '.' in date_str:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')


def is_dsum(notetype):
    if notetype is None:
        return False
    ntl = notetype.lower()
    return ('discharge summary' in ntl or 'discharge note' in ntl) and not 'nurs' in ntl


def label_io(mrn):
    notes_fn = os.path.join(notes_dir, mrn, 'notes.csv')
    invalid_accounts, valid_accounts = [], []
    if not os.path.exists(notes_fn):
        return 0, 0

    notes_df = pd.read_csv(notes_fn)
    nonzero_valid_visits = 'account' in notes_df.columns
    if not nonzero_valid_visits:
        return 0, 0

    notes_df.dropna(subset=['account'], inplace=True)
    notes_df.fillna({'note_type': 'N/A'}, inplace=True)
    notes_df['timestamp'] = notes_df['timestamp'].apply(str_to_dt)
    notes_df.sort_values('timestamp', inplace=True)
    notes_df.reset_index(inplace=True, drop=True)
    notes_df['is_target'] = notes_df['note_type'].apply(is_dsum)
    notes_n = notes_df.shape[0]
    is_source = [False] * notes_n
    accounts = notes_df['account'].unique().tolist()

    for account in accounts:
        note_account_df = notes_df[notes_df['account'] == account]
        dsums = note_account_df[note_account_df['is_target']]
        dsum_timestamp = None
        has_target = False
        if dsums.shape[0] > 0:
            dsum_timestamp = dsums['timestamp'].min()
            has_target = True
        for x, note_row in note_account_df.iterrows():
            note_row = note_row.to_dict()
            is_pre_dsum = False if dsum_timestamp is None else note_row['timestamp'] < dsum_timestamp
            ntl = note_row['note_type'].lower()
            dsum_related = 'discharge' in ntl
            is_source[x] = is_pre_dsum and not dsum_related
        is_valid = any(is_source) and has_target
        if is_valid:
            valid_accounts.append(str(account))
        else:
            invalid_accounts.append(str(account))
    notes_df['is_source'] = is_source
    notes_df.to_csv(notes_fn, index=False)
    with open(os.path.join(notes_dir, mrn, 'valid_accounts.txt'), 'w') as fd:
        fd.write('\n'.join(valid_accounts))
    with open(os.path.join(notes_dir, mrn, 'invalid_accounts.txt'), 'w') as fd:
        fd.write('\n'.join(invalid_accounts))

    return len(valid_accounts), len(invalid_accounts)


if __name__ == '__main__':
    mrns = os.listdir(notes_dir)
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    p = Pool()
    counts = p.map(label_io, mrns)
    end_time = time()

    minutes = (end_time - start_time) / 60.0
    round_factor = 0
    if minutes < 1:
        round_factor = 2
    print('Took {} minutes'.format(minutes, round(round_factor)))
    valid_account_n = sum([x[0] for x in counts])
    invalid_account_n = sum([x[1] for x in counts])
    print('{} visits are valid (have >= 1 dsum and >= 1 preceeding documents).'.format(valid_account_n))
    print('{} visits are invalid (have 0 dsum or 0 preceeding documents).'.format(invalid_account_n))
