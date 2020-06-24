from datetime import datetime
import os
from multiprocessing import Pool
import shutil

import pandas as pd
from tqdm import tqdm

visit_fn = '/nlp/projects/clinsum/inpatient_visits_reduced.csv'
notes_dir = '/nlp/projects/clinsum/notes_by_mrn'

visit_df = pd.read_csv(visit_fn)


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


def str_to_dt(str):
    return datetime.strptime(str.split(' ')[0], '%Y-%m-%d').date()


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
    mrn_visit_df = visit_df[visit_df['mrn'] == int(mrn)]
    mrn_visit_df['start_date'] = mrn_visit_df['admit_date_min'].apply(str_to_dt)
    mrn_visit_df['end_date'] = mrn_visit_df['discharge_date_max'].apply(str_to_dt)
    mrn_visit_df.sort_values('start_date', inplace=True)
    num_visits = mrn_visit_df.shape[0]
    assert num_visits > 0

    visit_records = mrn_visit_df.to_dict('records')
    valid_visit_records = remove_overlapping(visit_records)
    av = len(visit_records) - len(valid_visit_records)

    notes_fn = os.path.join(notes_dir, mrn, 'notes.csv')
    if os.path.exists(notes_fn) and len(valid_visit_records) > 0:
        notes_df = pd.read_csv(notes_fn)
        note_dates = notes_df['timestamp'].apply(str_to_dt).tolist()
        notes_df['account'] = list(map(lambda x: get_visit_id(x, visit_records), note_dates))
        notes_df.to_csv(notes_fn, index=False)
    else:
        shutil.rmtree(os.path.join(notes_dir, mrn))

    return av, len(valid_visit_records)


if __name__ == '__main__':
    mrns = os.listdir(notes_dir)
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    p = Pool()
    counts = p.map(join, mrns)
    end_time = time()

    minutes = (end_time - start_time) / 60.0

    ambiguous_visits = sum(map(lambda x: x[0], counts))
    clear_visits = sum(map(lambda x: x[1], counts))
    print('Visits identified={}. Ambiguous (near overlapping) visits removed={}'.format(clear_visits, ambiguous_visits))
