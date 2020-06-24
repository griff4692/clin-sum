from collections import defaultdict
from datetime import datetime
import glob
import os
import random
import re
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
MIN_DATETIME = datetime(2010, 1, 1)


HEADER_SEARCH_REGEX = r'^\s*([A-Z][A-z0-9/ ]+[A-z]):'


def str_to_dt(str):
    return datetime.strptime(str, '%Y-%m-%d-%H.%M.%S.%f')


def is_dsum(notetype):
    if notetype is None:
        return False
    ntl = notetype.lower()
    return ('discharge summary' in ntl or 'discharge note' in ntl) and not 'nurs' in ntl


def main(max=250):
    """
    Description: Loops through the avro folder and extracts notes for mrn list.
    """
    all_avro_fns = glob.glob(os.path.join(avro_fp, '*.avro'))
    med_code_cols = ['note_type', 'notetype_medcode', 'note_descname', 'loinc_medcode', 'loinc_descname', 'axis_name']
    med_code_df = pd.read_csv(med_code_fn, sep='\t', header=None, names=med_code_cols)
    med_code_df.dropna(subset=['note_type', 'notetype_medcode'], inplace=True)
    med_code_map = med_code_df.set_index('notetype_medcode').to_dict()['note_type']
    # cols = ['mrn', 'timestamp', 'med_code', 'note_type', 'title', 'fname', 'note_id', 'text']
    print('Extracting section header counts for dsums')

    section_counts = defaultdict(int)

    total, done = 0, False
    print('Starting to process avro files...')
    tmp_fd = open('tmp.txt', 'w')
    for i in tqdm(range(len(all_avro_fns))):
        fn = all_avro_fns[i]
        with open(fn, 'rb') as fd:
            for record in reader(fd):
                datetime = str_to_dt(record['TIME_STR_KEY'])
                content = record['TEXT']
                med_code = int(record['EVENT_CODE'])
                notetype = med_code_map.get(med_code, None)
                if datetime >= MIN_DATETIME and is_dsum(notetype):
                    header_list = re.findall(HEADER_SEARCH_REGEX, content, flags=re.M)
                    for header in header_list:
                        section_counts[header] += 1
                    total += 1
                    if total >= max:
                        done = True
                        break

                    dashes = '_' * 100
                    tmp_fd.write(content)
                    tmp_fd.write('\n' + dashes + '\n' + dashes + '\n' + dashes + '\n')
        if done:
            break

    sorted_items = list(sorted(section_counts.items(), key=lambda x: x[1], reverse=True))
    df = pd.DataFrame(sorted_items, columns=['section', 'count'])
    out_fn = 'section_counts.csv'
    print('Saving {} distinct sections with corpus frequency to {}'.format(df.shape[0], out_fn))
    df.to_csv(out_fn, index=False)


if __name__ == '__main__':
    main()
