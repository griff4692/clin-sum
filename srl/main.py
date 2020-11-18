from collections import defaultdict
import os
import re
from string import punctuation
import sys

import argparse
from fuzzysearch import find_near_matches
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from p_tqdm import p_uimap
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.extractive_fragments import DELIM as FRAG_DELIM
from preprocess.section_utils import resolve_course, paragraph_from_html, replace_paragraphs
from preprocess.tokenize_mrns import sent_segment
from preprocess.utils import *


MIN_FRAG_LEN = 5
FINDING_TUI = 'T033'


def get_disorders(df):
    return df[((df['sem_group'] == 'Disorders') & (df['tui'] != FINDING_TUI))]


def non_punc_tok_count(str):
    toks = str.split(' ')
    return len([t for t in toks if t not in punctuation])


def pack_ents(text, ent_df, paired_cuis, disorder_cuis):
    matches = []
    ent_df['source_lens'] = ent_df['source_value'].apply(lambda x: len(x))
    ent_df_records = ent_df.sort_values('source_lens', ascending=False).to_dict('records')
    for entity in ent_df_records:
        source_val = entity['source_value']
        tui = entity['tui']
        search_regex = re.escape(source_val)
        cui_name_nospace = '_'.join(entity['cui_name'].split(' '))
        cui = entity['cui']
        is_disorder = '1' if cui in disorder_cuis else '0'
        match = re.search(search_regex, text)
        if match is not None:
            s, e = match.start(), match.end()
            has_pair = '1' if cui in paired_cuis else '0'
            ent_str = '<u cui={} name={} tui={} has_pair={} dis={}> '.format(
                cui, cui_name_nospace, tui, has_pair, is_disorder) + text[s:e] + ' </u>'
            text = text[:s] + '<{}>'.format(len(matches)) + text[e:]
            matches.append(ent_str)

    dig_split = r'(<\d{1,3}>)'
    final_text = []
    for snippet in re.split(dig_split, text):
        if re.match(dig_split, snippet) is not None:
            # replace with corresponding match string
            idx = int(snippet[1:-1])
            final_text.append(matches[idx])
        else:
            final_text.append(snippet)

    packed_text = ''.join(final_text)
    return packed_text


def find_s_e(frag, text):
    fl = frag.lower()
    tl = text.lower()
    dist_range = list(range(5))
    for dist in dist_range:
        matches = find_near_matches(fl, tl, max_l_dist=dist)
        if len(matches) > 0:
            return [(m.start, m.end) for m in matches]
    return []


def pack_frags(paras, frags):
    n = len(paras)
    starts = [defaultdict(list) for _ in range(n)]
    frag_matches = defaultdict(set)

    for frag_idx, frag in enumerate(frags):
        for pidx, para in enumerate(paras):
            matches = find_s_e(frag, para)
            output = set([(m, pidx, frag_idx) for m in matches])
            frag_matches[frag].update(output)
            for match in matches:
                starts[pidx][match[0]] = [match[1], frag_idx, frag]

    packed_paras = []
    for p_idx, start in enumerate(starts):
        curr_idx = 0
        raw_para = paras[p_idx]
        packed_para = ''
        start_idxs = list(sorted(start.keys()))
        for start_idx in start_idxs:
            end_idx, frag_idx, _ = start[start_idx]
            packed_para += (raw_para[curr_idx:start_idx] + ' <f idx={}> '.format(frag_idx) +
                            raw_para[start_idx:end_idx] + ' </f> ')
            curr_idx = end_idx
        packed_para += raw_para[curr_idx:]
        packed_paras.append(packed_para)

    return ' <p> '.join(packed_paras)


def pack_example(example):
    account = example['account']
    mrn = example['mrn']
    frags = [] if type(example['fragments']) == float else example['fragments'].split(FRAG_DELIM)
    frag_lens = [non_punc_tok_count(frag) for frag in frags]
    long_frags = [frags[i] for i, frag_len in enumerate(frag_lens) if frag_len >= MIN_FRAG_LEN]
    long_frags = list(set(long_frags))
    source_paras = paragraph_from_html(example['source_str'])
    course_str = resolve_course(example['target_str'])
    target_paras = paragraph_from_html(course_str)
    packed_source = pack_frags(source_paras, long_frags)
    packed_target = pack_frags(target_paras, long_frags)
    ent_fn = os.path.join(out_dir, 'entity', 'entities', '{}_{}.csv'.format(mrn, account))

    ent_df = pd.read_csv(ent_fn)
    ent_df.dropna(subset=['cui', 'cui_name', 'source_value'], inplace=True)
    ent_source_df = ent_df[ent_df['is_source']]
    ent_target_df = ent_df[ent_df['is_target']]

    source_cuis = set(ent_source_df['cui'].tolist())
    target_cuis = set(ent_target_df['cui'].tolist())
    paired_cuis = target_cuis.intersection(source_cuis)

    source_disorder_cuis = set(get_disorders(ent_source_df)['cui'].unique().tolist())
    target_disorder_cuis = set(get_disorders(ent_target_df)['cui'].unique().tolist())

    full_packed_source = pack_ents(packed_source, ent_source_df, paired_cuis, source_disorder_cuis)
    full_packed_target = pack_ents(packed_target, ent_target_df, paired_cuis, target_disorder_cuis)

    srl_packed_source = replace_paragraphs(example['source_str'], full_packed_source.split(' <p> '))
    srl_packed_target = replace_paragraphs(course_str, full_packed_target.split(' <p> '))

    example['srl_packed_source'] = srl_packed_source
    example['srl_packed_target'] = srl_packed_target
    example['long_fragments'] = FRAG_DELIM.join(long_frags)

    return example


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to pack source and summary with entity and copy-paste fragments.')
    parser.add_argument('-mini', default=False, action='store_true')

    args = parser.parse_args()

    splits = ['validation', 'train']
    examples = get_records(split=splits, mini=args.mini).to_dict('records')
    n = len(examples)

    mini_str = '_small' if args.mini else ''
    examples_packed = list(p_uimap(pack_example, examples, num_cpus=0.8))
    out_fn = os.path.join(out_dir, 'srl_packed_examples{}.csv'.format(mini_str))
    print('Done! Now saving {} packed examples to {}'.format(len(examples_packed), out_fn))
    packed_examples_df = pd.DataFrame(examples_packed)
    packed_examples_df.to_csv(out_fn, index=False)
