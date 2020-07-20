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

from lexrank import STOPWORDS
from lexrank.algorithms.power_method import stationary_distribution
from lexrank.utils.text import tokenize
import numpy as np
import pandas as pd
from tqdm import tqdm

from cohort.constants import out_dir
from cohort.section_utils import pack_sentences, resolve_course, sents_from_html, sent_toks_from_html
from cohort.utils import get_mrn_status_df


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
    i, mrn = mrn
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

    if (i + 1) % 1000 == 0:
        print('Processed {} examples'.format(i + 1))
    return outputs


if __name__ == '__main__':
    splits_df = pd.read_csv(os.path.join(out_dir, 'splits.csv'))
    val_df = splits_df[splits_df['split'] == 'validation']
    val_mrns = val_df['mrn'].unique().tolist()
    val_examples = val_df[['mrn', 'account']].to_dict('records')
    pool = Pool()
    outputs = list(filter(None, pool.map(gen_summaries, enumerate(val_mrns))))
    outputs_flat = list(itertools.chain(*outputs))
    out_fn = os.path.join('predictions', 'lr_validation.csv')
    output_df = pd.DataFrame(outputs_flat)
    output_df.to_csv(out_fn, index=False)
