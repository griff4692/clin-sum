from collections import namedtuple
import itertools
import os
import pickle
import sys
import re
from time import time

import argparse
import numpy as np
import pandas as pd
from p_tqdm import p_uimap
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from evaluations.rouge import prepare_str_for_rouge, compute
from preprocess.constants import *
from preprocess.utils import *
from preprocess.section_utils import sents_from_html


MAX_SUMMARIES = 25000
MAX_TARGET_SENTS = 100


Example = namedtuple(
    'Example',
    [
        'mrn', 'account', 'curr_sum_sents', 'source_sents_lexranks', 'candidate_source_sents', 'curr_rouge',
        'target_rouges', 'target_sents'
    ]
)


def extraction_is_keep(str):
    MIN_WORD_CT = 3
    any_alpha = re.search('[a-zA-Z]', str) is not None
    toks = str.split(' ')
    long_enough = len(toks) >= MIN_WORD_CT
    return any_alpha and long_enough


def get_score(metric, recall_weight=1.0):
    if recall_weight == 1.0:
        return metric.fmeasure
    p, r = metric.precision, metric.recall
    num = p * r
    beta = recall_weight ** 2
    denom = max(beta * p + r, 1e-5)
    return (1 + beta) * num / denom


def generate_samples(row):
    single_extraction_examples = []

    source_sents, source_lrs = sents_from_html(row['spacy_source_toks_packed'], convert_lower=True, extract_lr=True)
    target_sents = sents_from_html(row['spacy_target_toks'], convert_lower=True)
    target = ' '.join(target_sents)
    target_no_stop = prepare_str_for_rouge(target)

    source_sents_no_stop = list(map(prepare_str_for_rouge, source_sents))
    # remove 1-2 word sentences (too many of them) and most are not necessary for BHC
    keep_idxs = [idx for idx, s in enumerate(source_sents_no_stop) if extraction_is_keep(s)]
    source_sents_no_stop_filt = [source_sents_no_stop[idx] for idx in keep_idxs]
    source_lrs_filt = [source_lrs[idx] for idx in keep_idxs]
    source_n = len(keep_idxs)
    curr_sum_sents = []
    curr_rouge = 0.0
    included_sent_idxs = set()
    max_target_n = min(source_n, len(target_sents), MAX_TARGET_SENTS)
    references = [target_no_stop for _ in range(source_n)]
    for _ in range(max_target_n):
        curr_sum = ' '.join(curr_sum_sents).strip() + ' '
        predictions = [(curr_sum + s).strip() for s in source_sents_no_stop_filt]
        outputs = compute(
            predictions=predictions, references=references, rouge_types=['rouge1', 'rouge2'], use_agregator=False)
        scores = np.array(list(map(
            lambda x: (get_score(x[0]) + get_score(x[1])) / 2.0,
            zip(outputs['rouge1'], outputs['rouge2']))
        ))
        if len(included_sent_idxs) > 0:
            scores[list(included_sent_idxs)] = float('-inf')
        max_idx = np.argmax(scores)
        score = scores[max_idx]
        if score <= curr_rouge:
            break

        eligible_scores = []
        eligible_source_sents = []
        eligible_lrs = []
        for i in range(len(scores)):
            if i not in included_sent_idxs:
                eligible_scores.append(scores[i])
                eligible_source_sents.append(source_sents_no_stop_filt[i])
                eligible_lrs.append(source_lrs_filt[i])
        # Example
        example = Example(
            mrn=row['account'],
            account=row['account'],
            curr_sum_sents=curr_sum_sents.copy(),
            candidate_source_sents=eligible_source_sents,
            source_sents_lexranks=source_lrs_filt,
            curr_rouge=curr_rouge,
            target_rouges=eligible_scores,
            target_sents=target_sents,
        )
        single_extraction_examples.append(example)
        curr_rouge = score
        curr_sum_sents.append(source_sents_no_stop_filt[int(max_idx)])
        included_sent_idxs.add(max_idx)
    return single_extraction_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate greedy labels for single-step extraction task.')
    parser.add_argument('-mini', default=False, action='store_true')
    parser.add_argument('-single_proc', default=False, action='store_true')

    args = parser.parse_args()
    mini_str = '_mini' if args.mini else ''
    in_fn = os.path.join(out_dir, 'full_examples{}.csv'.format(mini_str))

    print('Loading full examples from {}'.format(in_fn))
    examples_df = pd.read_csv(in_fn)

    splits_fn = os.path.join(out_dir, 'splits.csv')
    splits_df = pd.read_csv(splits_fn)[['mrn', 'split']].drop_duplicates(subset=['mrn'])
    df = examples_df.merge(splits_df, on='mrn')
    assert len(df) == len(examples_df)

    types = ['train', 'validation', 'test']
    for type in types:
        type_df = df[df['split'] == type]
        if len(type_df) > MAX_SUMMARIES:
            print('Shrinking from {} to {}'.format(len(type_df), MAX_SUMMARIES))
            type_df = type_df.sample(n=MAX_SUMMARIES, replace=False)
        type_examples = type_df.to_dict('records')
        n = len(type_examples)
        print('Processing {} examples for {} set'.format(n, type))
        if args.single_proc:
            single_extraction_examples = list(tqdm(map(generate_samples, type_examples), total=n))
        else:
            single_extraction_examples = list(p_uimap(generate_samples, type_examples))

        output = list(itertools.chain(*single_extraction_examples))
        out_n = len(output)
        out_fn = os.path.join(out_dir, 'single_extraction_labels_{}{}.pk'.format(type, mini_str))
        print('Saving {} labeled single step extraction samples to {}'.format(out_n, out_fn))
        with open(out_fn, 'wb') as fd:
            pickle.dump(output, fd)
