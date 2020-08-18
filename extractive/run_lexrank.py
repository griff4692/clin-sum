from collections import Counter, defaultdict
from functools import partial
import itertools
import json
import math
from multiprocessing import Pool
import os
from string import punctuation
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import argparse
from lexrank import STOPWORDS
from lexrank.algorithms.power_method import stationary_distribution
from lexrank.utils.text import tokenize
import numpy as np
import pandas as pd
from p_tqdm import p_imap

from preprocess.constants import out_dir
from preprocess.section_utils import pack_sentences, resolve_course, sents_from_html, sent_toks_from_html
from preprocess.utils import get_mrn_status_df


def top_k_sents(packed_str, k=12, preserve_order=True):
    sents, lr_scores = sents_from_html(packed_str, convert_lower=False, extract_lr=True)
    n = len(sents)
    k = min(n, k)
    top_k = np.argsort(-lr_scores)[:k]
    if preserve_order:
        top_k = set(top_k)
        return [sents[i] for i in range(n) if i in top_k]
    else:
        return [sents[i] for i in top_k]


def gen_summaries(mrn):
    example_fn = os.path.join(out_dir, 'mrn', str(mrn), 'examples.csv')
    example_df = pd.read_csv(example_fn)
    if 'spacy_source_toks_packed' not in example_df.columns:
        return False

    records = example_df.to_dict('records')
    outputs = []
    for record in records:
        top_sents = top_k_sents(record['spacy_source_toks_packed'])
        target_toks = sent_toks_from_html(resolve_course(record['spacy_target_toks']), convert_lower=False)
        summary = ' '.join(top_sents)
        reference = ' '.join(target_toks)

        ref_len = len(target_toks)
        sum_len = len(summary.split(' '))

        outputs.append({
            'account': record['account'],
            'mrn': record['mrn'],
            'prediction': summary,
            'reference': reference,
            'sum_len': sum_len,
            'ref_len': ref_len,
        })

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate Lexrank predictions')
    parser.add_argument('--max_n', default=-1, type=int)

    args = parser.parse_args()

    splits_df = pd.read_csv(os.path.join(out_dir, 'splits.csv'))
    val_df = splits_df[splits_df['split'] == 'validation']
    val_mrns = val_df['mrn'].unique().tolist()

    if args.max_n > 0:
        np.random.seed(1992)
        val_mrns = np.random.choice(val_mrns, size=args.max_n, replace=False)

    outputs = list(filter(None, p_imap(gen_summaries, val_mrns)))
    outputs_flat = list(itertools.chain(*outputs))
    out_fn = os.path.join(out_dir, 'predictions', 'lr_validation.csv')
    output_df = pd.DataFrame(outputs_flat)
    output_df.to_csv(out_fn, index=False)
