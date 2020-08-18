from collections import Counter, defaultdict
from functools import partial
import itertools
import json
import math
from multiprocessing import Pool
import os
import pickle
from string import punctuation
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import argparse
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english')).union(set(list(punctuation)))
import numpy as np
import pandas as pd
from p_tqdm import p_uimap

from preprocess.constants import out_dir
from preprocess.section_utils import resolve_course, sents_from_html, sent_toks_from_html
from preprocess.utils import get_mrn_status_df


def gen_query(toks, max_q=20):
    tok_set = set(toks)
    tok_set -= STOPWORDS
    filter_toks = [t for t in list(tok_set) if not np.char.isnumeric(t)]
    if len(filter_toks) > 0:
        end_idx = min(max_q, len(filter_toks))
        return filter_toks[:end_idx]

    end_idx = min(max_q, len(toks))
    return toks[:end_idx]


def gen_summaries(mrn):
    example_fn = os.path.join(out_dir, 'mrn', str(mrn), 'examples.csv')
    example_df = pd.read_csv(example_fn)
    records = example_df.to_dict('records')
    outputs = []
    for record in records:
        target_sents = sents_from_html(resolve_course(record['spacy_target_toks']), convert_lower=True)
        summary_sents = list(map(
            lambda sent: bm25.get_top_n(gen_query(sent.split(' ')), corpus, n=1)[0],
            target_sents
        ))

        n = len(target_sents)

        reference = ' <s> '.join(target_sents).strip()
        summary = ' <s> '.join(summary_sents).strip()
        ref_len = len(reference.split(' ')) - n  # subtract pseudo sentence tokens
        sum_len = len(summary.split(' ')) - n  # subtract pseudo sentence tokens

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
    parser = argparse.ArgumentParser('Script to generate retrieval (BM-25) predictions')
    parser.add_argument('--max_n', default=-1, type=int)

    args = parser.parse_args()

    splits_df = pd.read_csv(os.path.join(out_dir, 'splits.csv'))
    val_df = splits_df[splits_df['split'] == 'validation']
    val_mrns = val_df['mrn'].unique().tolist()

    if args.max_n > 0:
        np.random.seed(1992)
        val_mrns = np.random.choice(val_mrns, size=args.max_n, replace=False)

    print('Loading BM25...')
    bm25_fn = os.path.join(out_dir, 'bm25.pk')
    with open(bm25_fn, 'rb') as fd:
        bm25 = pickle.load(fd)

    print('Loading original corpus (train sentences) for which BM25 is has indexed...')
    train_sents_fn = os.path.join(out_dir, 'train_sents.csv')
    corpus = pd.read_csv(train_sents_fn).sents.tolist()

    print('Let\'s retrieve!')
    outputs = list(filter(None, p_uimap(gen_summaries, val_mrns)))
    outputs_flat = list(itertools.chain(*outputs))
    out_fn = os.path.join(out_dir, 'predictions', 'retrieval_validation.csv')
    output_df = pd.DataFrame(outputs_flat)
    output_df.to_csv(out_fn, index=False)