from datetime import datetime
from functools import partial
import os
import re
import sys
from time import time

import pandas as pd
pd.options.mode.chained_assignment = None
from p_tqdm import p_imap

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.utils import *
from preprocess.section_utils import MIN_TARGET_LEN, extract_hospital_course, clean, sectionize


def stringify(x):
    if x == NULL_STR:
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
        clean_formatted_str = sectionize(clean(record['text']).strip())
        if clean_formatted_str in seen:
            continue
        seen.add(clean_formatted_str)
        if len(clean_formatted_str) > 0:
            source_str += '<d note_id={}> {} </d> '.format(record['note_id'], clean_formatted_str)
    source_str = source_str.strip()
    if len(source_str) >= MIN_SOURCE_LEN:
        return '<e account={}> {} </e>'.format(str(account), source_str)
    return ''


def generate_examples(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    valid_accounts_fn = os.path.join(mrn_dir, 'valid_accounts.csv')
    notes_fn = os.path.join(mrn_dir, 'notes.csv')
    examples_fn = os.path.join(mrn_dir, 'examples.csv')

    valid_accounts = pd.read_csv(valid_accounts_fn).dropna()['account'].astype('str').tolist()

    notes_df = pd.read_csv(notes_fn)
    notes_df = notes_df.fillna(NULL_STR)
    notes_df['account'] = notes_df['account'].apply(stringify)

    examples, sources, targets = [], [], []
    for account in valid_accounts:
        account_notes = notes_df[notes_df['account'] == account]
        source_df = account_notes[account_notes['is_source']]
        target_df = account_notes[account_notes['is_target']]
        assert source_df.shape[0] > 0 and target_df.shape[0] > 0
        source_note_str = process_source(source_df.to_dict('records'), account)
        target_note_str = process_target(target_df.to_dict('records'), account)
        if len(source_note_str) > len(target_note_str) > 0:
            examples.append(
                (mrn, account, len(source_note_str), len(target_note_str), source_note_str, target_note_str))

    if len(examples) > 0:
        df = pd.DataFrame(examples, columns=['mrn', 'account', 'source_len', 'target_len', 'source_str', 'target_str'])
        df.to_csv(examples_fn, index=False)
        return 1
    return 0


if __name__ == '__main__':
    mrn_status_df, mrn_valid_idxs, mrns = get_mrn_status_df('valid_account')
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    statuses = p_imap(generate_examples, mrns)
    update_mrn_status_df(mrn_status_df, list(statuses), mrn_valid_idxs, 'valid_example')
    duration(start_time)
