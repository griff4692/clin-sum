from collections import defaultdict
from datetime import datetime
import glob
import os
import random
import shutil
import string
import sys
import time

from fastavro import reader
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from tqdm import tqdm


avro_fp = '/nlp/cdw/discovery_request_1342/notes_avro/all_docs_201406190000/'
out_dir = '/nlp/projects/clinsum/notes_by_mrn'
med_code_fn = '/nlp/cdw/note_medcodes/notetype_loinc.txt'

MIN_DOC_LEN = 25


def _save(row, cols, fn):
    df = pd.DataFrame(row, columns=cols)
    if os.path.exists(fn):
        df.to_csv(fn, mode='a', header=False, index=False)
    else:
        df.to_csv(fn, index=False)


def str_to_dt(str):
    return datetime.strptime(str, '%Y-%m-%d-%H.%M.%S.%f')


def main(max=None):
    """
    Description: Loops through the avro folder and extracts notes for mrn list.
    """
    all_avro_fns = glob.glob(os.path.join(avro_fp, '*.avro'))
    med_code_cols = ['note_type', 'notetype_medcode', 'note_descname', 'loinc_medcode', 'loinc_descname', 'axis_name']
    med_code_df = pd.read_csv(med_code_fn, sep='\t', header=None, names=med_code_cols)
    med_code_df.dropna(subset=['note_type', 'notetype_medcode'], inplace=True)
    med_code_map = med_code_df.set_index('notetype_medcode').to_dict()['note_type']
    cols = ['mrn', 'timestamp', 'med_code', 'note_type', 'title', 'fname', 'note_id', 'text']
    mrn_filter = pd.read_csv('/nlp/projects/clinsum/inpatient_visits_reduced.csv')['mrn'].unique()
    mrn_filter = set(map(str, mrn_filter))
    print('Collecting all notes for {} mrns'.format(len(mrn_filter)))
    N = len(all_avro_fns)

    if os.path.exists(out_dir):
        print('Clearing out {}'.format(out_dir))
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    print('Creating sub directories for each mrn')
    for mrn in mrn_filter:
        os.mkdir(os.path.join(out_dir, mrn))

    total, done = 0, False
    print('Starting to process avro files...')
    for n in tqdm(range(N)):
        fn = all_avro_fns[n]
        with open(fn, 'rb') as fd:
            for record in reader(fd):
                date_time = str_to_dt(record['TIME_STR_KEY'])
                title = record['TITLE']
                content = record['TEXT']
                mrn = record['MRN']
                out_fn = os.path.join(out_dir, mrn, 'notes.csv')
                note_id = "_".join([mrn, record['TIME_STR_KEY'], title])
                med_code = int(record['EVENT_CODE'])

                note_type = med_code_map.get(med_code, None)

                if mrn not in mrn_filter:
                    continue

                row = {
                    'mrn': [mrn],
                    'timestamp': [date_time],
                    'med_code': [med_code],
                    'note_type': [note_type],
                    'title': [title],
                    'fname': [fn],
                    'note_id': [note_id],
                    'text': content
                }

                if content is not None and len(content) >= MIN_DOC_LEN:
                    _save(row, cols, out_fn)
                    total += 1
                    if total % 1000 == 0:
                        print('Added {} notes'.format(total))
                    if max is not None and total >= max:
                        done = True
            if done:
                break
        if done:
            break


if __name__ == '__main__':
    main()
