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
from evaluations.rouge import prepare_str_for_rouge, compute
from preprocess.constants import *
from preprocess.utils import *
from preprocess.section_utils import sents_from_html

MAX_TARGET_SENTS = 100  # For skipping examples when generating dataset
MAX_SOURCE_SENTS = 2000

MAX_SUMMARIES = 17500
MAX_SUM_SENTS = 50

# Don't include dubious training examples.  Not a clear enough signal which sentence to pick
MIN_ROUGE_IMPROVEMENT = 0.01  # if relative gain is less than this, don't create training set
MIN_ROUGE_DIFFERENTIAL = 0.01  # if difference between worst scoring rouge sentence and best is less than this


def extraction_is_keep(str, target_toks, no_match_keep_prob=0.33):
    MIN_WORD_CT = 3
    any_alpha = re.search('[a-zA-Z]', str) is not None
    toks = str.split(' ')
    any_matches = False
    for tok in toks:
        if tok in target_toks:
            any_matches = True
            break
    long_enough = len(toks) >= MIN_WORD_CT
    any_matches = any_matches or random() <= no_match_keep_prob
    return any_alpha and long_enough and any_matches


def get_score(metric, recall_weight=1.0):
    if recall_weight == 1.0:
        return metric.fmeasure
    p, r = metric.precision, metric.recall
    num = p * r
    beta = recall_weight ** 2
    denom = max(beta * p + r, 1e-5)
    return (1 + beta) * num / denom


def compute_no_match_keep_prob(source_n, is_test):
    if is_test:
        return 1.0
    if source_n < 100:
        return 1.0
    elif source_n < 250:
        return 0.75
    elif source_n < 500:
        return 0.5
    elif source_n < 1000:
        return 0.25
    else:
        return 0.1


def generate_samples(row):
    """
    :param row:
    :return:
    """
    rouge_types = ['rouge1', 'rouge2']
    single_extraction_examples = []

    source_sents, source_lrs = sents_from_html(row['spacy_source_toks_packed'], convert_lower=True, extract_lr=True)
    target_sents = sents_from_html(row['spacy_target_toks'], convert_lower=True)
    target_n = len(target_sents)

    if target_n > MAX_TARGET_SENTS:
        return []

    target = ' '.join(target_sents)
    target_no_stop = prepare_str_for_rouge(target)
    target_toks = set(target_no_stop.split(' '))

    source_sents_no_stop = list(map(prepare_str_for_rouge, source_sents))
    dup_idxs = set()
    seen = set()
    for idx, source_sent in enumerate(source_sents_no_stop):
        if source_sent in seen:
            dup_idxs.add(idx)
        else:
            seen.add(source_sent)
    # remove duplicate sentences and 1-2 word sentences (too many of them) and most are not necessary for BHC
    keep_idxs = [
        idx for idx, s in enumerate(source_sents_no_stop) if extraction_is_keep(
            s, target_toks, no_match_keep_prob=compute_no_match_keep_prob(len(source_sents), type == 'test')
        ) and idx not in dup_idxs
    ]
    source_sents_no_stop_filt = [source_sents_no_stop[idx] for idx in keep_idxs]
    source_lrs_filt = [source_lrs[idx] for idx in keep_idxs]
    source_n = len(keep_idxs)

    if source_n < target_n or source_n > MAX_SOURCE_SENTS:
        return []

    curr_sum_sents = []
    curr_rouge = 0.0
    included_sent_idxs = set()
    max_sum_n = min(source_n, len(target_sents), MAX_SUM_SENTS)
    references = [target_no_stop for _ in range(source_n)]

    for gen_idx in range(max_sum_n):
        curr_sum = ' '.join(curr_sum_sents).strip() + ' '
        predictions = [(curr_sum + s).strip() for s in source_sents_no_stop_filt]
        outputs = compute(predictions=predictions, references=references, rouge_types=rouge_types, use_aggregator=False)
        scores = np.array(
            [sum([outputs[t][i].fmeasure for t in rouge_types]) / float(len(rouge_types)) for i in range(source_n)])
        scores_pos_mask = scores.copy()
        if len(included_sent_idxs) > 0:
            scores[list(included_sent_idxs)] = float('-inf')
            scores_pos_mask[list(included_sent_idxs)] = float('inf')
        max_idx = np.argmax(scores)
        max_score = scores[max_idx]
        min_score = scores_pos_mask.min()
        max_differential = max_score - min_score
        max_gain = max_score - curr_rouge
        if max_gain < MIN_ROUGE_IMPROVEMENT or max_differential < MIN_ROUGE_DIFFERENTIAL:
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
        example = {
            'mrn': row['account'],
            'account': row['account'],
            'curr_sum_sents': curr_sum_sents.copy(),
            'candidate_source_sents': eligible_source_sents,
            'source_sents_lexranks': eligible_lrs,
            'curr_rouge': curr_rouge,
            'target_rouges': eligible_scores,
            'target_sents': target_sents,
        }

        single_extraction_examples.append(example)
        curr_rouge = max_score
        curr_sum_sents.append(source_sents_no_stop_filt[int(max_idx)])
        included_sent_idxs.add(max_idx)
    return single_extraction_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate greedy labels for single-step extraction task.')
    parser.add_argument('-mini', default=False, action='store_true')
    parser.add_argument('-single_proc', default=False, action='store_true')

    args = parser.parse_args()

    mini_str = '_mini' if args.mini else ''
    types = ['validation', 'train']
    for type in types:
        type_df = get_records(type=type, mini=args.mini)
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
        out_fn = os.path.join(out_dir, 'single_extraction_labels_{}{}.json'.format(type, mini_str))
        print('Saving {} labeled single step extraction samples to {}'.format(out_n, out_fn))
        with open(out_fn, 'w') as fd:
            json.dump(output, fd)