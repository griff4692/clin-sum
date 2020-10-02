import itertools
import os
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import argparse
import numpy as np
import pandas as pd
from p_tqdm import p_uimap

from preprocess.constants import out_dir
from preprocess.section_utils import resolve_course, sents_from_html, sent_toks_from_html
from preprocess.utils import *


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


def gen_summaries(record):
    top_sents = top_k_sents(record['spacy_source_toks_packed'])
    target_toks = sent_toks_from_html(record['spacy_target_toks'], convert_lower=False)
    summary = ' '.join(top_sents)
    reference = ' '.join(target_toks)
    ref_len = len(target_toks)
    sum_len = len(summary.split(' '))

    return {
        'account': record['account'],
        'mrn': record['mrn'],
        'prediction': summary,
        'reference': reference,
        'sum_len': sum_len,
        'ref_len': ref_len,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate Lexrank predictions')
    parser.add_argument('--max_n', default=-1, type=int)

    args = parser.parse_args()

    validation_df = get_records(type='validation')
    validation_records = validation_df.to_dict('records')

    if args.max_n > 0:
        np.random.seed(1992)
        validation_records = np.random.choice(validation_records, size=args.max_n, replace=False)

    outputs = list(filter(None, p_uimap(gen_summaries, validation_records, num_cpus=0.8)))
    out_fn = os.path.join(out_dir, 'predictions', 'lr_validation.csv')
    output_df = pd.DataFrame(outputs)
    output_df.to_csv(out_fn, index=False)
