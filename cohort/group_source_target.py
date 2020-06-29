from datetime import datetime
from functools import partial
import os
from multiprocessing import Pool, Manager
import re
from time import time

import pandas as pd
pd.options.mode.chained_assignment = None
from tqdm import tqdm

from extract_course import MIN_TARGET_LEN, extract_hospital_course, clean


EXAMPLE_DELIM = '|' * 5
MIN_SOURCE_LEN = 100
HEADER_SEARCH_REGEX = r'(?:^|\s{4,}|\n)[\d.#]{0,4}\s*([A-Z][A-z0-9/ ]+[A-z]:|[A-Z0-9/ ]+\n)'
SEP_REGEX = r'\.\s|\n{2,}|^\s{0,}\d{1,2}\s{0,}[-).]\s{1,}'
notes_dir = '/nlp/projects/clinsum/notes_by_mrn'


def stringify(x):
    if x == 'N/A':
        return x
    return str(int(float(x)))


def process_target(target_records, account):
    target_str = ''
    seen = set()
    for record in target_records:
        course_str = extract_hospital_course(clean(record['text'])).strip()
        if course_str in seen:
            continue
        seen.add(course_str)
        if len(course_str) > 0:
            target_str += '<d note_id={}> {} </d> '.format(record['note_id'], course_str)
    target_str = target_str.strip()
    if len(target_str) >= MIN_TARGET_LEN:
        return '<e account={}> {} </e>'.format(str(account), target_str)
    return ''


def process_source(source_records, account):
    source_str = ''
    seen = set()
    for record in source_records:
        clean_str = clean(record['text']).strip()
        if clean_str in seen:
            continue
        seen.add(clean_str)
        if len(clean_str) > 0:
            source_str += '<d note_id={}> {} </d> '.format(record['note_id'], clean_str)
    source_str = source_str.strip()
    if len(source_str) >= MIN_SOURCE_LEN:
        return '<e account={}> {} </e>'.format(str(account), source_str)
    return ''


def generate_examples(mrn, valid_counter=None, invalid_counter=None, lock=None):
    valid_accounts_fn = os.path.join(notes_dir, mrn, 'valid_accounts.csv')
    notes_fn = os.path.join(notes_dir, mrn, 'notes.csv')
    source_fn = os.path.join(notes_dir, mrn, 'data.source')
    target_fn = os.path.join(notes_dir, mrn, 'data.target')
    examples_fn = os.path.join(notes_dir, mrn, 'examples.txt')
    if not os.path.exists(valid_accounts_fn):
        return 0, 0

    valid_accounts = pd.read_csv(valid_accounts_fn).dropna()['account'].astype('str').tolist()
    notes_df = pd.read_csv(notes_fn)

    # TODO will be deprecated
    notes_df.fillna({'account': 'N/A'}, inplace=True)
    # Might need to keep this
    notes_df['account'] = notes_df['account'].apply(stringify)

    examples, sources, targets = [], [], []
    for account in valid_accounts:
        account_notes = notes_df[notes_df['account'] == account]
        source_df = account_notes[account_notes['is_source']]
        dsum_df = account_notes[account_notes['is_dsum']]
        if source_df.shape[0] == 0:
            print('MRN={} Account={} has no source documents'.format(mrn, account))
            return
        if dsum_df.shape[0] == 0:
            print('MRN={} Account={} has no target documents'.format(mrn, account))
            return
        # if passes put # assert source_df.shape[0] > 0 and target_df.shape[0] > 0

        source_note_str = process_source(source_df.to_dict('records'), account)
        target_note_str = process_target(dsum_df.to_dict('records'), account)
        if len(source_note_str) > 0 and len(target_note_str) > 0:
            sources.append(source_note_str)
            targets.append(target_note_str)
            examples.append(account)

            with lock:
                valid_counter.value += 1
                if valid_counter.value % 10000 == 0:
                    print('Dsums with Hospital Course={}. {} without.'.format(
                        valid_counter.value, invalid_counter.value))
        else:
            with lock:
                invalid_counter.value += 1

    with open(source_fn, 'w') as fd:
        fd.write(EXAMPLE_DELIM.join(sources))

    with open(target_fn, 'w') as fd:
        fd.write(EXAMPLE_DELIM.join(targets))

    with open(examples_fn, 'w') as fd:
        fd.write('\n'.join(examples))


if __name__ == '__main__':
    mrns = os.listdir(notes_dir)
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    with Manager() as manager:
        valid_counter = manager.Value('i', 0)
        invalid_counter = manager.Value('i', 0)
        lock = manager.Lock()
        pool = Pool()  # By default pool will size depending on cores available
        pool.map(
            partial(generate_examples, valid_counter=valid_counter, invalid_counter=invalid_counter, lock=lock),
            mrns
        )
        pool.close()
        pool.join()

        print('Visits with Hospital Course in dsum={}.'.format(valid_counter.value))
        print('Visits without Hospital Course in dsum={}.'.format(invalid_counter.value))
    end_time = time()
    minutes = (end_time - start_time) / 60.0
    round_factor = 0
    if minutes < 1:
        round_factor = 2
    print('Took {} minutes'.format(minutes, round(round_factor)))
