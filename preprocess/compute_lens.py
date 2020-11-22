from collections import defaultdict
import itertools
import json
import os
from random import random
import sys
import re

import argparse
import numpy as np
from p_tqdm import p_uimap
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.utils import *
from preprocess.section_utils import sents_from_html, sent_toks_from_html


def generate_counts(example):
    source = example['spacy_source_toks']
    target = example['spacy_target_toks']
    source_sents = sents_from_html(source)
    target_sents = sents_from_html(target)
    source_toks = sent_toks_from_html(source)
    target_toks = sent_toks_from_html(target)

    source_sent_lens, target_sent_lens = [], []
    for source_sent in source_sents:
        source_sent_lens.append(len(source_sent.split(' ')))

    for target_sent in target_sents:
        target_sent_lens.append(len(target_sent.split(' ')))

    num_docs = len(re.findall(r'd note_id', source))
    return ({
        'mrn': example['mrn'],
        'account': example['account'],
        'source_toks': len(source_toks),
        'target_toks': len(target_toks),
        'source_sents': len(source_sents),
        'target_sents': len(target_sents),
        'source_docs': num_docs,
        'target_docs': 1,
    }, {'source_sent_lens': source_sent_lens, 'target_sent_lens': target_sent_lens})


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to compute dataset statistics.')

    args = parser.parse_args()

    in_fn = os.path.join(out_dir, 'full_examples_no_trunc.csv')
    print('Loading data from {}'.format(in_fn))
    df = pd.read_csv(in_fn)
    print('Loaded {} distinct visits'.format(len(df)))
    outputs = list(p_uimap(generate_counts, df.to_dict('records'), num_cpus=0.8))

    counts = [output[0] for output in outputs]
    source_sent_lens = np.array(list(itertools.chain(*[output[1]['source_sent_lens'] for output in outputs])))
    target_sent_lens = np.array(list(itertools.chain(*[output[1]['target_sent_lens'] for output in outputs])))

    print('Source sentence length. Mean={}. STD={}.'.format(np.mean(source_sent_lens), np.std(source_sent_lens)))
    print('Target sentence length. Mean={}. STD={}.'.format(np.mean(target_sent_lens), np.std(target_sent_lens)))

    count_df = pd.DataFrame(output)
    count_df['total_docs'] = count_df['source_docs'].apply(lambda x: x + 1)
    count_fn = 'stats/counts_no_trunc.csv'
    count_df.to_csv(count_fn, index=False)

    print('Sum...')
    print(count_df.sum())

    print('Mean...')
    print(count_df.mean())

    print('Median...')
    print(count_df.median())

    print('Min...')
    print(count_df.min())

    print('Max...')
    print(count_df.max())
