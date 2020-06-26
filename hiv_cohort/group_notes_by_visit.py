from datetime import datetime
import os
from multiprocessing import Pool
import shutil
from time import time

import pandas as pd
pd.options.mode.chained_assignment = None
from tqdm import tqdm

visit_fn = '/nlp/projects/clinsum/inpatient_visits.csv'
notes_dir = '/nlp/projects/clinsum/notes_by_mrn'


def separate_by_col(df, col):
    df_groups = df.groupby(col)
    return {str(k): pd.DataFrame(df.loc[v]).to_dict('records') for k, v in df_groups.groups.items()}


def str_to_d(date_str):
    return datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d').date()


def str_to_dt(date_str):
    if len(date_str.split(' ')) == 1:
        return datetime.strptime(date_str, '%Y-%m-%d')
    if '.' in date_str:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')


visit_df = pd.read_csv(visit_fn)
print('Preprocessing visit dataframe...')
visit_df['start_date'] = visit_df['admit_date_min'].apply(str_to_d)
visit_df['end_date'] = visit_df['discharge_date_max'].apply(str_to_d)
mrn_visit_records = separate_by_col(visit_df, 'mrn')


def get_visit_id(date, visit_records):
    """
    :param date:
    :param visit_records:
    :return:
    """
    for idx in range(len(visit_records)):
        r = visit_records[idx]
        if r['start_date'] <= date <= r['end_date']:
            return r['account']
    return None


def is_dsum(notetype):
    if notetype is None:
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
    mrn_visit_record = mrn_visit_records[mrn]
    num_visits = len(mrn_visit_record)
    assert num_visits > 0

    non_overlapping_visit_records = remove_overlapping(mrn_visit_record)
    mrn_dir = os.path.join(notes_dir, mrn)
    notes_fn = os.path.join(mrn_dir, 'notes.csv')

    notes_df = pd.read_csv(notes_fn)
    note_dates = notes_df['timestamp'].apply(str_to_d).tolist()
    notes_df['account'] = list(map(lambda x: get_visit_id(x, non_overlapping_visit_records), note_dates))

    # Needs to be associated with a visit, have content, and have a timestamp
    notes_df.dropna(subset=['account', 'text', 'timestamp'], inplace=True)
    notes_df.fillna({'note_type': 'N/A'}, inplace=True)
    notes_df['timestamp'] = notes_df['timestamp'].apply(str_to_dt)
    notes_df.sort_values('timestamp', inplace=True)
    notes_df.reset_index(inplace=True, drop=True)
    notes_df['is_target'] = notes_df['note_type'].apply(is_dsum)
    notes_n = notes_df.shape[0]
    is_source = [False] * notes_n
    accounts = notes_df['account'].unique().tolist()

    valid_account_n, invalid_account_n = 0, 0
    for account in accounts:
        note_account_df = notes_df[notes_df['account'] == account]
        dsums = note_account_df[note_account_df['is_target']]
        n_dsums = dsums.shape[0]
        has_target = n_dsums >= 1
        dsum_timestamp = dsums['timestamp'].min() if has_target else None
        for x, note_row in note_account_df.iterrows():
            note_row = note_row.to_dict()
            is_pre_dsum = False if dsum_timestamp is None else note_row['timestamp'] < dsum_timestamp
            ntl = note_row['note_type'].lower()
            dsum_related = 'discharge' in ntl
            is_source[x] = is_pre_dsum and not dsum_related
        is_valid = any(is_source) and has_target
        if is_valid:
            valid_account_n += 1
        else:
            invalid_account_n += 1
    notes_df['is_source'] = is_source
    if valid_account_n == 0:
        print('Removing mrn directory --> {}'.format(mrn_dir))
        shutil.rmtree(mrn_dir)
    else:
        notes_df.to_csv(notes_fn, index=False)

    return valid_account_n, invalid_account_n


if __name__ == '__main__':
    mrns = os.listdir(notes_dir)
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    p = Pool()
    counts = p.map(join, mrns)
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
