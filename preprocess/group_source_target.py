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
from preprocess.section_utils import (
    HTML_REGEX, MIN_TARGET_LEN, extract_hospital_course, clean, sectionize, resolve_course, paragraph_toks_from_html
)


def stringify(x):
    if x == NULL_STR:
        return x
    return str(int(float(x)))


def process_target(target_records, account):
    target_str = ''
    seen = set()
    for record in target_records:
        course_str, _ = extract_hospital_course(clean(record['text']))
        course_str = course_str.strip()
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
        clean_formatted_str = sectionize(clean(record['text']).strip(), exclude_course=True)
        if clean_formatted_str in seen:
            continue
        seen.add(clean_formatted_str)
        if len(clean_formatted_str) > 0:
            source_str += '<d note_id={}> {} </d> '.format(record['note_id'], clean_formatted_str)
    source_str = source_str.strip()

    split_text = re.split(HTML_REGEX, source_str)
    is_tag = list(map(lambda x: re.search(HTML_REGEX, x) is not None, split_text))
    remove_idxs = []
    seen_paras = set()
    for idx, (st, it) in enumerate(zip(split_text, is_tag)):
        st = st.strip()
        if st.startswith('<p') and it:
            para = split_text[idx + 1].strip()
            if para in seen_paras:
                assert split_text[idx + 2] == '</p>'
                remove_idxs += [idx, idx + 1, idx + 2]
            seen_paras.add(para)

    final_pieces = []
    for idx, st in enumerate(split_text):
        if idx in remove_idxs:
            continue
        final_pieces.append(st)

    source_str_no_dup_paras = ''.join(final_pieces)
    ratio = len(source_str_no_dup_paras) / float(len(source_str))
    assert ratio <= 1
    if len(source_str_no_dup_paras) >= MIN_SOURCE_LEN:
        return '<e account={}> {} </e>'.format(str(account), source_str_no_dup_paras)
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
    notes_df.drop_duplicates(subset=['text'], inplace=True)

    seen_targets = set()

    examples, sources, targets = [], [], []
    for account in valid_accounts:
        account_notes = notes_df[notes_df['account'] == account]
        source_df = account_notes[account_notes['is_source']]
        target_df = account_notes[account_notes['is_target']]
        source_n, target_n = len(source_df), len(target_df)
        if len(source_df) == 0 or len(target_df) == 0:
            print('MRN={}. Account={}. Source Docs={}. Target docs={}. Skipping...'.format(
                mrn, account, source_n, target_n
            ))
            continue
        assert source_df.shape[0] > 0 and target_df.shape[0] > 0
        source_note_str = process_source(source_df.to_dict('records'), account)
        target_note_str = process_target(target_df.to_dict('records'), account)
        if len(source_note_str) > len(target_note_str) > 0:
            raw_target = ' '.join(paragraph_toks_from_html(resolve_course(target_note_str)))
            if raw_target in seen_targets:
                print('Duplicate hospital course.  MRN={}. Account={}.'.format(mrn, account))
                continue
            seen_targets.add(raw_target)
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
    statuses = list(p_imap(generate_examples, mrns, num_cpus=0.8))
    print('{} out of {} are valid'.format(sum(statuses), len(statuses)))
    update_mrn_status_df(mrn_status_df, list(statuses), mrn_valid_idxs, 'valid_example')
