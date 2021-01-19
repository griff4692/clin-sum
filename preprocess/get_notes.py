from collections import defaultdict
from datetime import datetime
from itertools import chain
from functools import partial
import glob
from itertools import chain
from multiprocessing import Lock, Manager, Pool, Value
import os
import random
import re
import shutil
import string
import sys
from time import time

from fastavro import reader
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from p_tqdm import p_uimap

from constants import *
from utils import *

SAVE_COLS = ['mrn', 'timestamp', 'med_code', 'note_type', 'title', 'fname', 'note_id', 'text']
mrn_dir = os.path.join(out_dir, 'mrn')

VALID_EVENT_STATUSES = {'A', 'E', 'F', 'P'}


def has_relevant_title(title, nan_val):
    if type(title) == float or title is None:
        return nan_val

    tl = title.lower()
    # keep_frags = [
    #     'admission',
    #     'admit',
    #     'assessment',
    #     'consult',
    #     'discharge',
    #     'free text',
    #     'progress',
    # ]

    remove_frags = ['nurse', 'nursing', 'student', 'checklist']

    # sr = r'({})'.format('|'.join(keep_frags))
    # rel = re.search(sr, tl) is not None
    nr = r'({})'.format('|'.join(remove_frags))
    is_remove = re.search(nr, tl) is not None
    # return rel and not is_remove
    return not is_remove


def save(mrn_rows, suffix):
    n = 0
    fn = None
    notes_dir = os.path.join(out_dir, 'notes')
    for mrn, rows in mrn_rows.items():
        fn = rows[0]['fname']
        n += len(rows)
        df = pd.DataFrame(rows)
        df = df.fillna(NULL_STR)
        notes_fn = os.path.join(notes_dir, '{}_{}.csv'.format(mrn, suffix))
        df.to_csv(notes_fn, index=False)
    print('Saving {} notes for {} MRNs from {}'.format(n, len(mrn_rows), fn))


def worker(fn, med_code_map, mrn_filter):
    suffix = fn.split('/')[-1].split('.')[0]
    mrn_rows = defaultdict(list)
    with open(fn, 'rb') as fd:
        for record in reader(fd):
            if not record['EVENT_STATUS'] in VALID_EVENT_STATUSES:
                continue
            date_time = str_to_dt(record['TIME_STR_KEY'], trunc=True)
            update_time = datetime.fromtimestamp(record['UPDATE_TIME'] / 1e3).replace(minute=0, second=0, microsecond=0)
            if MIN_YEAR <= date_time.year <= MAX_YEAR:
                title = record['TITLE']
                content = record['TEXT']
                mrn = record['MRN']
                note_id = '_'.join([mrn, record['TIME_STR_KEY'], title])
                med_code = int(record['EVENT_CODE'])
                note_type = med_code_map.get(med_code, None)
                keep_nt = has_relevant_title(note_type, nan_val=True)
                keep_title = has_relevant_title(title, nan_val=False)

                if keep_nt and keep_title and mrn in mrn_filter:
                    row = {
                        'mrn': mrn,
                        'created_time': date_time,
                        'update_time': update_time,
                        'event_status': record['EVENT_STATUS'],
                        'med_code': med_code,
                        'note_type': note_type,
                        'title': title,
                        'fname': fn,
                        'note_id': note_id,
                        'text': content
                    }
                    if content is not None and len(content) >= MIN_DOC_LEN:
                        mrn_rows[mrn].append(row)
    if len(mrn_rows) > 0:
        save(mrn_rows, suffix)

    return list(mrn_rows.keys())


def save_mrn(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn')
    notes_dir = os.path.join(out_dir, 'notes')
    pattern = os.path.join(notes_dir, str(mrn) + '*.csv')
    mrn_fns = glob.glob(pattern)
    assert len(mrn_fns) > 0
    df = []
    for mrn_fn in mrn_fns:
        df.append(pd.read_csv(mrn_fn))
    df = pd.concat(df)

    # sort by UPDATE_TIME
    # Take note with latest update time for all dups (dups = same note_id or text)
    df.sort_values('update_time', inplace=True)
    df.drop_duplicates(subset=['note_id'], keep='last', inplace=True)
    df.sort_values('created_time', inplace=True)

    output_dir = os.path.join(mrn_dir, str(mrn))
    os.mkdir(output_dir)

    output_fn = os.path.join(output_dir, 'notes.csv')
    df.to_csv(output_fn, index=False)
    return len(df)


if __name__ == '__main__':
    """
    Description: Loops through the avro folder and extracts notes for mrn list.
    """
    all_avro_fns = sorted(glob.glob(os.path.join(avro_fp, '*.avro')))

    mrn_dir = os.path.join(out_dir, 'mrn')
    notes_dir = os.path.join(out_dir, 'notes')
    if os.path.exists(notes_dir):
        print('Clearing out {}'.format(notes_dir))
        shutil.rmtree(notes_dir)

    if os.path.exists(mrn_dir):
        print('Clearing out {}'.format(mrn_dir))
        shutil.rmtree(mrn_dir)

    print('Making a fresh dir at {}'.format(notes_dir))
    os.mkdir(notes_dir)

    print('Making a fresh dir at {}'.format(mrn_dir))
    os.mkdir(mrn_dir)

    med_code_cols = ['note_type', 'notetype_medcode', 'note_descname', 'loinc_medcode', 'loinc_descname', 'axis_name']
    med_code_df = pd.read_csv(med_code_fn, sep='\t', header=None, names=med_code_cols)
    med_code_df.dropna(subset=['note_type', 'notetype_medcode'], inplace=True)
    med_code_map = med_code_df.set_index('notetype_medcode').to_dict()['note_type']

    mrn_status_df, mrn_valid_idxs, all_mrns = get_mrn_status_df('has_visit')
    print('Collecting notes for {} mrns...'.format(len(all_mrns)))

    valid_mrns = list(p_uimap(partial(
        worker, med_code_map=med_code_map, mrn_filter=set(all_mrns)), all_avro_fns, num_cpus=0.8))
    valid_mrns_uniq = set(list(chain(*valid_mrns)))

    print('Saving MRN status for each of the original MRNs...')
    note_statuses = list(map(lambda mrn: 1 if mrn in valid_mrns_uniq else 0, all_mrns))

    num_notes = sum(list(p_uimap(save_mrn, list(valid_mrns_uniq))))
    update_mrn_status_df(mrn_status_df, note_statuses, mrn_valid_idxs, 'note_status')
    print('Saved {} notes for {} mrns'.format(num_notes, len(valid_mrns_uniq)))
