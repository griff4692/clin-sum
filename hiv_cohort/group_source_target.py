from datetime import datetime
import os
from multiprocessing import Pool
import re
from time import time

import pandas as pd
from tqdm import tqdm

notes_dir = '/nlp/projects/clinsum/notes_by_mrn'
MIN_SOURCE_LEN = 100
MIN_TARGET_LEN = 10

SECTION_MAPS = {
    'hospital_course': r'hospital course'
}

HEADER_SEARCH_REGEX = r'(?:^|\s{4,}|\n)[\d.#]{0,4}\s*([A-Z][A-z0-9/ ]+[A-z]:|[A-Z0-9/ ]+\n)'
SEP_REGEX = r'\.\s|\n{2,}|^\s{0,}\d{1,2}\s{0,}[-).]\s{1,}'


def extract_hospital_course(text):
    section_regex = r'{}'.format('|'.join(matches))
    if not partial_match:
        section_regex = r'^({})$'.format(section_regex)
    sectioned_text = list(filter(lambda x: len(x.strip()) > 0, re.split(HEADER_SEARCH_REGEX, text, flags=re.M)))
    is_header_arr = list(map(lambda x: re.match(HEADER_SEARCH_REGEX, x, re.M) is not None, sectioned_text))
    is_relevant_section = list(
        map(lambda x: re.match(section_regex, x.strip(':').lower(), re.M) is not None, sectioned_text))

    relevant_section_idxs = [i for i, x in enumerate(is_relevant_section) if x]
    for i in relevant_section_idxs:
        assert is_header_arr[i]
    n = len(relevant_section_idxs)
    if n == 0:
        return None

    str = ''
    for sec_idx in relevant_section_idxs:
        str += ' <s> {} </s> <p> {} </p>'.format(sectioned_text[sec_idx].strip(), sectioned_text[sec_idx + 1].strip())
    return str.strip()


def process_target(target_df, section=SECTION_MAPS['hospital_course']):
    return None


def process_source(source_df):
    return None


def get_sections(mrn):
    valid_accounts_fn = os.path.join(notes_dir, mrn, 'valid_accounts.txt')
    notes_fn = os.path.join(notes_dir, mrn, 'notes.csv')
    source_fn = os.path.join(notes_dir, mrn, 'data.source')
    target_fn = os.path.join(notes_dir, mrn, 'data.target')
    valid_fn = os.path.join(notes_dir, mrn, 'valid_accounts_inc_course.txt')

    valid_accounts = []
    with open(valid_accounts_fn, 'r') as fd:
        for line in fd:
            if len(line) > 0:
                valid_accounts.append(line)

    notes_df = pd.read_csv(notes_fn)
    actual_valid_accounts, source_notes, target_notes = [], [], []
    for account in valid_accounts:
        account_notes = notes_df[notes_df['account'] == int(account)]
        source_note_str = process_source(account_notes[account_notes['is_source']])
        target_note_str = process_target(account_notes[account_notes['is_target']])

        raise

        if len(source_note_str) >= MIN_SOURCE_LEN and len(target_note_str) >= MIN_TARGET_LEN:
            source_notes.append(source_note_str)
            target_notes.append(target_note_str)
            actual_valid_accounts.append(str(account))

    with open(source_fn, 'w') as fd:
        fd.write('\n'.join(source_notes))

    with open(target_fn, 'w') as fd:
        fd.write('\n'.join(target))

    with open(valid_fn, 'w') as fd:
        fd.write('\n'.join(actual_valid_accounts))


if __name__ == '__main__':
    mrns = os.listdir(notes_dir)
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    p = Pool()
    counts = p.map(get_sections, mrns)
    end_time = time()

    minutes = (end_time - start_time) / 60.0
    round_factor = 0
    if minutes < 1:
        round_factor = 2
    print('Took {} minutes'.format(minutes, round(round_factor)))
