from datetime import datetime
import os
from multiprocessing import Pool
import re
from time import time

import pandas as pd
pd.options.mode.chained_assignment = None
from tqdm import tqdm

notes_dir = '/nlp/projects/clinsum/notes_by_mrn'
MIN_SOURCE_LEN = 100
from extract_course import MIN_TARGET_LEN, extract_hospital_course, clean


HEADER_SEARCH_REGEX = r'(?:^|\s{4,}|\n)[\d.#]{0,4}\s*([A-Z][A-z0-9/ ]+[A-z]:|[A-Z0-9/ ]+\n)'
SEP_REGEX = r'\.\s|\n{2,}|^\s{0,}\d{1,2}\s{0,}[-).]\s{1,}'


def stringify(x):
    if x is None:
        return 'N/A'
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


def generate_examples(mrn):
    valid_accounts_fn = os.path.join(notes_dir, mrn, 'valid_accounts.txt')
    notes_fn = os.path.join(notes_dir, mrn, 'notes.csv')
    source_fn = os.path.join(notes_dir, mrn, 'data.source')
    target_fn = os.path.join(notes_dir, mrn, 'data.target')
    examples_fn = os.path.join(notes_dir, mrn, 'examples.txt')

    valid_accounts = []
    with open(valid_accounts_fn, 'r') as fd:
        for line in fd:
            if len(line) > 0:
                valid_accounts.append(line.strip())

    notes_df = pd.read_csv(notes_fn)
    # TODO will be deprecated because we do this filtering earlier
    notes_df.dropna(subset=['text'], inplace=True)
    examples, sources, targets = [], [], []
    for account in valid_accounts:
        account_str = stringify(account)
        account_notes = notes_df[notes_df['account'] == int(account_str)]
        source_df = account_notes[account_notes['is_source']]
        target_df = account_notes[account_notes['is_target']]
        source_note_str = process_source(source_df.to_dict('records'), account_str)
        target_note_str = process_target(target_df.to_dict('records'), account_str)

        note_types = source_df['note_type'].tolist() + target_df['note_type'].tolist()
        for nt in note_types:
            if type(nt) == float:
                continue
            ntl = nt.lower()
            if 'nursing' in ntl or 'nurse' in ntl:
                raise Exception(nt)

        if len(source_note_str) > 0 and len(target_note_str) > 0:
            sources.append(source_note_str)
            targets.append(target_note_str)
            examples.append(account_str)

    with open(source_fn, 'w') as fd:
        fd.write('\n'.join(sources))

    with open(target_fn, 'w') as fd:
        fd.write('\n'.join(targets))

    with open(examples_fn, 'w') as fd:
        fd.write('\n'.join(examples))

    return len(examples), len(valid_accounts)


if __name__ == '__main__':
    mrns = os.listdir(notes_dir)
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    p = Pool()
    counts = p.map(generate_examples, mrns)
    end_time = time()

    minutes = (end_time - start_time) / 60.0
    round_factor = 0
    if minutes < 1:
        round_factor = 2
    print('Took {} minutes'.format(minutes, round(round_factor)))

    ex = sum(map(lambda x: x[0], counts))
    total = sum(map(lambda x: x[1], counts))
    print('{} out of {} possible examples generated'.format(ex, total))
