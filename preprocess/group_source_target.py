from datetime import datetime
from functools import partial
import itertools
import os
import re
import shutil
import sys
from time import time

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from p_tqdm import p_imap

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.utils import *
from preprocess.section_utils import (
    HTML_REGEX, MIN_TARGET_LEN, extract_hospital_course, clean, sectionize, resolve_course, paragraph_toks_from_html
)


MAX_SOURCE_CHARACTER_COUNTS = 500000


def stringify(x):
    if x == NULL_STR:
        return x
    return str(int(float(x)))


def process_target(target_records, mrn, account):
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
        return '<e mrn={} account={}> {} </e>'.format(str(account), str(mrn), target_str)
    return ''


def process_source(source_records, mrn, account):
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
        return '<e mrn={} account={}> {} </e>'.format(str(mrn), str(account), source_str_no_dup_paras)
    return ''


def gen_clamp_input(formatted_str):
    split_text = re.split(HTML_REGEX, formatted_str)
    is_tag = list(map(lambda x: re.search(HTML_REGEX, x) is not None, split_text))
    clamp_metadata = []
    clamp_str = ''
    for idx, (st, it) in enumerate(zip(split_text, is_tag)):
        st = st.strip()
        if st.startswith('<p') and it:
            para = re.sub(r'\s+', ' ', split_text[idx + 1].strip()).strip()
            clamp_str += para + '\n'
        elif st.startswith('<h') and it:
            header = re.sub(r'\s+', ' ', split_text[idx + 1].strip()).strip()
            if not header.endswith(':'):
                header += ':'
            clamp_str += header.upper() + '\n'
        elif st.startswith('<d') and it:
            note_id = split_text[idx].split('note_id=')[-1].strip('>')
            clamp_metadata.append({'start': len(clamp_str), 'note_id': note_id})

    return clamp_str, clamp_metadata


def dump_clamp(clamp_input, clamp_metadata, mrn, account, suffix):
    base_fn = '{}_{}_{}'.format(str(mrn), str(account), suffix)
    input_fn = os.path.join(out_dir, 'clamp', 'input', base_fn + '.txt')
    with open(input_fn, 'w') as fd:
        fd.write(clamp_input)
    metadata_fn = os.path.join(out_dir, 'clamp', 'metadata', base_fn + '.csv')
    clamp_metadata_df = pd.DataFrame(clamp_metadata)
    clamp_metadata_df.to_csv(metadata_fn, index=False)


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

    too_big_ct = 0

    lens = []
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
        source_note_str = process_source(source_df.to_dict('records'), mrn, account)
        target_note_str = process_target(target_df.to_dict('records'), mrn, account)
        if len(source_note_str) > len(target_note_str) > 0:
            raw_target = ' '.join(paragraph_toks_from_html(resolve_course(target_note_str)))
            if raw_target in seen_targets:
                print('Duplicate hospital course.  MRN={}. Account={}.'.format(mrn, account))
                continue

            if max(len(source_note_str), len(target_note_str)) > MAX_SOURCE_CHARACTER_COUNTS:
                too_big_ct += 1
                continue

            seen_targets.add(raw_target)
            examples.append(
                (mrn, account, len(source_note_str), len(target_note_str), source_note_str, target_note_str))

            # generate CLAMP input here
            clamp_source_input, clamp_source_metadata = gen_clamp_input(source_note_str)
            clamp_target_input, clamp_target_metadata = gen_clamp_input(resolve_course(target_note_str))

            dump_clamp(clamp_source_input, clamp_source_metadata, mrn, account, 'source')
            dump_clamp(clamp_target_input, clamp_target_metadata, mrn, account, 'target')

            lens.append(len(source_note_str))

    if len(examples) > 0:
        df = pd.DataFrame(examples, columns=['mrn', 'account', 'source_len', 'target_len', 'source_str', 'target_str'])
        df.to_csv(examples_fn, index=False)
        return 1, lens, too_big_ct
    return 0, lens, too_big_ct


if __name__ == '__main__':
    clamp_dir = os.path.join(out_dir, 'clamp')
    # if os.path.exists(clamp_dir):
    #     print('Recursively removing {}'.format(clamp_dir))
    #     shutil.rmtree(clamp_dir)
    # os.mkdir(clamp_dir)
    # os.mkdir(os.path.join(clamp_dir, 'metadata'))
    # os.mkdir(os.path.join(clamp_dir, 'input'))
    # os.mkdir(os.path.join(clamp_dir, 'tmp'))
    # os.mkdir(os.path.join(clamp_dir, 'output'))
    mrn_status_df, mrn_valid_idxs, mrns = get_mrn_status_df('valid_account')
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    outputs = list(p_imap(generate_examples, mrns, num_cpus=0.8))
    statuses = [o[0] for o in outputs]
    source_lens = np.array(list(itertools.chain(*[o[1] for o in outputs])))
    too_big_ct = sum([o[2] for o in outputs])
    print('Average source character count is {}.  Max allowed is {}. {} examples deemed too big to include.'.format(
        source_lens.mean(), MAX_SOURCE_CHARACTER_COUNTS, too_big_ct))
    print('{} out of {} are valid'.format(sum(statuses), len(statuses)))
    update_mrn_status_df(mrn_status_df, list(statuses), mrn_valid_idxs, 'valid_example')
