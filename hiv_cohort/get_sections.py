from datetime import datetime
import os

import pandas as pd
from tqdm import tqdm

visit_fn = '/nlp/projects/clinsum/inpatient_visits_reduced.csv'
notes_dir = '/nlp/projects/clinsum/notes_by_mrn'


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


if __name__ == '__main__':
    ct = 0

    with open('tmp.txt', 'w') as fd:
        mrns = os.listdir(notes_dir)
        for mrn in mrns:
            notes_fn = os.path.join(notes_dir, mrn, 'notes.csv')
            if os.path.exists(notes_fn):
                notes_df = pd.read_csv(notes_fn)
                notes_df.dropna(subset=[
                    'note_type'
                ], inplace=True)
                dsums = notes_df[notes_df['note_type'].astype(str).str.lower().str.contains('discharge summary')]
                for text, nt, title in zip(dsums['text'], dsums['note_type'], dsums['title']):
                    fd.write(nt + ' --> ' + title + '\n')
                    fd.write(text)
                    fd.write('\n' + ('_' * 100) + '\n')
                    ct += 1

                    if ct > 25:
                        raise
