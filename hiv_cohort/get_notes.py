from collections import defaultdict
from datetime import datetime
from functools import partial
import glob
from itertools import chain
from multiprocessing import Lock, Manager, Pool, Queue, Value
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


avro_fp = '/nlp/cdw/discovery_request_1342/notes_avro/all_docs_201406190000/'
out_dir = '/nlp/projects/clinsum/notes_by_mrn'
med_code_fn = '/nlp/cdw/note_medcodes/notetype_loinc.txt'

MIN_DOC_LEN = 25
SAVE_COLS = ['mrn', 'timestamp', 'med_code', 'note_type', 'title', 'fname', 'note_id', 'text']
BLACKLIST_SEARCH = r'nurse|nursing'


class SimpleCounter:
    def __init__(self):
        self.value = 0


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
        mrn_fn = os.path.join(out_dir, mrn)
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
            title = record['TITLE']
            content = record['TEXT']
            mrn = record['MRN']
            note_id = '_'.join([mrn, record['TIME_STR_KEY'], title])
            med_code = int(record['EVENT_CODE'])
            note_type = med_code_map.get(med_code, None)
            keep_nt = note_type is None or re.search(BLACKLIST_SEARCH, note_type.lower()) is None

            if keep_nt and mrn in mrn_filter:
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
        if lock is None:
            save(mrn_rows, note_counter)
        else:
            with lock:
                save(mrn_rows, note_counter)

    return list(mrn_rows.keys())


def main():
    """
    Description: Loops through the avro folder and extracts notes for mrn list.
    """
    all_avro_fns = sorted(glob.glob(os.path.join(avro_fp, '*.avro')))
    pool_arg = sys.argv[1]
    max_n = None if len(sys.argv) == 2 else int(sys.argv[2])
    if max_n is not None:
        np.random.seed(1992)
        np.random.shuffle(all_avro_fns)
        all_avro_fns = all_avro_fns[:max_n]
        print('Truncated to {} random avro files (for debugging)'.format(max_n))

    if os.path.exists(out_dir):
        print('Clearing out {}'.format(out_dir))
        shutil.rmtree(out_dir)
    print('Making a fresh dir at {}'.format(out_dir))
    os.mkdir(out_dir)

    med_code_cols = ['note_type', 'notetype_medcode', 'note_descname', 'loinc_medcode', 'loinc_descname', 'axis_name']
    med_code_df = pd.read_csv(med_code_fn, sep='\t', header=None, names=med_code_cols)
    med_code_df.dropna(subset=['note_type', 'notetype_medcode'], inplace=True)
    med_code_map = med_code_df.set_index('notetype_medcode').to_dict()['note_type']
    mrn_filter = set(pd.read_csv('/nlp/projects/clinsum/inpatient_visits.csv')['mrn'].unique().astype('str'))
    print('Collecting notes for {} mrns...'.format(len(mrn_filter)))

    start_time = time()

    if pool_arg == 'pool':
        with Manager() as manager:
            note_counter = manager.Value('i', 0)
            lock = manager.Lock()
            pool = Pool()  # By default pool will size depending on cores available
            mrns = list(pool.map(partial(
                worker, note_counter=note_counter, lock=lock, med_code_map=med_code_map, mrn_filter=mrn_filter),
                all_avro_fns
            ))
            pool.close()
            pool.join()
            mrns = set(list(chain.from_iterable(mrns)))
            mrn_count = len(mrns)
            print('Saved {} notes for {} patients'.format(note_counter.value, mrn_count))
    elif pool_arg == 'linear':
        note_counter = SimpleCounter()
        mrns = []
        for fn in all_avro_fns:
            result = worker(fn, note_counter=note_counter, lock=None, med_code_map=med_code_map, mrn_filter=mrn_filter)
            mrns.append(result)
        mrns = set(list(chain.from_iterable(mrns)))
        mrn_count = len(mrns)
        print('Saved {} notes for {} patients'.format(note_counter.value, mrn_count))
    else:
        raise Exception('Did\'nt recognize pool arg={}'.format(pool_arg))
    end_time = time()
    minutes = (end_time - start_time) / 60.0
    round_factor = 0
    if minutes < 1:
        round_factor = 2
    print('Took {} minutes'.format(minutes, round(round_factor)))
    assert mrn_count == len(os.listdir(out_dir))


if __name__ == '__main__':
    main()
