from collections import defaultdict
from datetime import datetime
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
from tqdm import tqdm

from constants import *
from utils import *

SAVE_COLS = ['mrn', 'timestamp', 'med_code', 'note_type', 'title', 'fname', 'note_id', 'text']
mrn_dir = os.path.join(out_dir, 'mrn')


def has_relevant_title(title, nan_val):
    if type(title) == float or title is None:
        return nan_val

    tl = title.lower()
    keep_frags = [
        'admission',
        'admit',
        'assessment',
        'consult',
        'discharge',
        'free text',
        'progress',
    ]

    remove_frags = ['nurse', 'nursing']

    sr = r'({})'.format('|'.join(keep_frags))
    rel = re.search(sr, tl) is not None
    nr = r'({})'.format('|'.join(remove_frags))
    is_remove = re.search(nr, tl) is not None
    return rel and not is_remove


def save(mrn_rows, note_counter):
    n = 0
    fn = None
    for mrn, rows in mrn_rows.items():
        fn = rows[0]['fname']
        n += len(rows)
        df = pd.DataFrame(rows, columns=SAVE_COLS)
        df.drop_duplicates(subset=[
            'note_id'
        ], inplace=True)
        df = df.fillna(NULL_STR)
        mrn_fn = os.path.join(mrn_dir, mrn)
        if not os.path.exists(mrn_fn):
            os.mkdir(mrn_fn)
        notes_fn = os.path.join(mrn_fn, 'notes.csv')
        if os.path.exists(notes_fn):
            df.to_csv(notes_fn, mode='a', header=False, index=False)
        else:
            df.to_csv(notes_fn, index=False)
    note_counter.value += n
    print('Saving {} notes for {} MRNs from {}'.format(n, len(mrn_rows), fn))


def str_to_dt(str):
    return datetime.strptime(str, '%Y-%m-%d-%H.%M.%S.%f')


def worker(fn, note_counter, lock, med_code_map, mrn_filter):
    mrn_rows = defaultdict(list)
    with open(fn, 'rb') as fd:
        for record in reader(fd):
            date_time = str_to_dt(record['TIME_STR_KEY'])
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
                        'timestamp': date_time,
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
        with lock:
            save(mrn_rows, note_counter)

    return list(mrn_rows.keys())


def main():
    """
    Description: Loops through the avro folder and extracts notes for mrn list.
    """
    all_avro_fns = sorted(glob.glob(os.path.join(avro_fp, '*.avro')))
    force = sys.argv[1] and sys.argv[1] == 'force'
    max_n = None if len(sys.argv) == 2 else int(sys.argv[2])
    if max_n is not None:
        np.random.seed(1992)
        np.random.shuffle(all_avro_fns)
        all_avro_fns = all_avro_fns[:max_n]
        print('Truncated to {} random avro files (for debugging)'.format(max_n))

    if os.path.exists(mrn_dir):
        if force:
            print('Clearing out {}'.format(mrn_dir))
            shutil.rmtree(mrn_dir)
        else:
            raise Exception('Either run with \'force\' flag or clear out {}'.format(mrn_dir))
    print('Making a fresh dir at {}'.format(mrn_dir))
    os.mkdir(mrn_dir)

    med_code_cols = ['note_type', 'notetype_medcode', 'note_descname', 'loinc_medcode', 'loinc_descname', 'axis_name']
    med_code_df = pd.read_csv(med_code_fn, sep='\t', header=None, names=med_code_cols)
    med_code_df.dropna(subset=['note_type', 'notetype_medcode'], inplace=True)
    med_code_map = med_code_df.set_index('notetype_medcode').to_dict()['note_type']

    mrn_status_df, mrn_valid_idxs, mrns = get_mrn_status_df('has_visit')
    print('Collecting notes for {} mrns...'.format(len(mrns)))

    start_time = time()

    with Manager() as manager:
        note_counter = manager.Value('i', 0)
        lock = manager.Lock()
        pool = Pool()  # By default pool will size depending on cores available
        mrns_w_notes = list(pool.map(partial(
            worker, note_counter=note_counter, lock=lock, med_code_map=med_code_map, mrn_filter=mrns),
            all_avro_fns
        ))
        pool.close()
        pool.join()
        mrns_w_notes = set(list(chain.from_iterable(mrns_w_notes)))
        mrn_count = len(mrns_w_notes)
        print('Saved {} notes for {} patients'.format(note_counter.value, mrn_count))

    print('Saving MRN status for each of the original MRNs...')
    note_statuses = list(map(lambda mrn: 1 if mrn in mrns_w_notes else 0, mrns))
    update_mrn_status_df(mrn_status_df, note_statuses, mrn_valid_idxs, 'note_status')

    duration(start_time)


if __name__ == '__main__':
    main()
