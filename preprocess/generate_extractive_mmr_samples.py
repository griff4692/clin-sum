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
from evaluations.rouge import prepare_str_for_rouge, compute
from preprocess.constants import *
from preprocess.utils import *
from preprocess.section_utils import sents_from_html

MAX_TARGET_SENTS = 100  # For skipping examples when generating dataset
MAX_SOURCE_SENTS = 2000

MAX_SUMMARIES = 100000
MAX_SUM_SENTS = 50

# Don't include dubious training examples.  Not a clear enough signal which sentence to pick
MIN_ROUGE_IMPROVEMENT = 0.02  # if relative gain is less than this, don't create training set
MIN_ROUGE_DIFFERENTIAL = 0.01  # if difference between worst scoring rouge sentence and best is less than this

# min word count to be considered for extractive summarization (remove pseudo-sentences with dates or just poor split)
MIN_WORD_CT = 3


def extraction_is_keep(str, target_toks, no_match_keep_prob=0.33):
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
    rouge_diffs = defaultdict(float)
    rouge_gains = defaultdict(float)
    rouge_fulls = defaultdict(float)

    source_sents = sents_from_html(row['spacy_source_toks'], convert_lower=True)
    target_sents = sents_from_html(row['spacy_target_toks'], convert_lower=True)
    target_n = len(target_sents)

    if not eval_mode and target_n > MAX_TARGET_SENTS:
        return [], rouge_diffs, rouge_gains, rouge_fulls

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
    should_keep_all = eval_mode or type == 'test'
    keep_idxs = [
        idx for idx, s in enumerate(source_sents_no_stop) if extraction_is_keep(
            s, target_toks, no_match_keep_prob=compute_no_match_keep_prob(len(source_sents), should_keep_all)
        ) and idx not in dup_idxs
    ]
    source_sents_no_stop_filt = [source_sents_no_stop[idx] for idx in keep_idxs]
    source_sents_filt = [source_sents[idx] for idx in keep_idxs]
    source_n = len(keep_idxs)

    if not should_keep_all and (source_n < target_n or source_n > MAX_SOURCE_SENTS):
        return [], rouge_diffs, rouge_gains, rouge_fulls

    curr_sum_sents = []
    curr_rouge = 0.0
    included_sent_idxs = set()
    max_sum_n = min(source_n, len(target_sents), MAX_SUM_SENTS)
    if eval_mode:
        max_sum_n = 1
    references = [target_no_stop for _ in range(source_n)]

    for gen_idx in range(max_sum_n):
        curr_sum = prepare_str_for_rouge(' '.join(curr_sum_sents).strip() + ' ')
        predictions = [(curr_sum + s).strip() for s in source_sents_no_stop_filt]
        outputs = compute(predictions=predictions, references=references, rouge_types=rouge_types, use_aggregator=False)
        scores = np.array(
            [sum([outputs[t][i].fmeasure for t in rouge_types]) / float(len(rouge_types)) for i in range(source_n)])
        scores_pos_mask = scores.copy()
        if len(included_sent_idxs) > 0:
            scores[list(included_sent_idxs)] = float('-inf')
            scores_pos_mask[list(included_sent_idxs)] = float('inf')
        max_idx = int(np.argmax(scores))
        max_score = scores[max_idx]
        assert max_idx not in included_sent_idxs
        min_score = scores_pos_mask.min()
        max_differential = max_score - min_score
        max_gain = max_score - curr_rouge

        valid_scores = []
        for score in scores:
            if score > -1:
                valid_scores.append(score)
        rouge_diffs[gen_idx] = max_differential
        rouge_gains[gen_idx] = max_score - np.mean(valid_scores)
        rouge_fulls[gen_idx] = max_score
        if max_gain < MIN_ROUGE_IMPROVEMENT or max_differential < MIN_ROUGE_DIFFERENTIAL:
            break

        eligible_scores = []
        eligible_source_sents = []
        for i in range(len(scores)):
            if i not in included_sent_idxs:
                eligible_scores.append(scores[i])
                eligible_source_sents.append(source_sents_filt[i])

        # Example
        example = {
            'mrn': row['mrn'],
            'account': row['account'],
            'curr_sum_sents': curr_sum_sents.copy(),
            'candidate_source_sents': eligible_source_sents,
            'curr_rouge': curr_rouge,
            'target_rouges': eligible_scores,
            'target_sents': target_sents,
        }

        single_extraction_examples.append(example)
        curr_rouge = max_score
        curr_sum_sents.append(source_sents_filt[max_idx])
        included_sent_idxs.add(max_idx)
    assert len(curr_sum_sents) == len(set(curr_sum_sents))
    return single_extraction_examples, rouge_diffs, rouge_gains, rouge_fulls


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate greedy labels for single-step extraction task.')
    parser.add_argument('-mini', default=False, action='store_true')
    parser.add_argument('-single_proc', default=False, action='store_true')
    parser.add_argument('-eval_mode', default=False, action='store_true')

    args = parser.parse_args()

    eval_mode = args.eval_mode
    mini_str = '_mini' if args.mini else ''
    eval_str = '_eval' if args.eval_mode else ''
    types = ['validation', 'train']
    for type in types:
        print('Getting records for {} set'.format(type))
        type_df = get_records(split=type, mini=args.mini)
        if not eval_mode and len(type_df) > MAX_SUMMARIES:
            print('Shrinking from {} to {}'.format(len(type_df), MAX_SUMMARIES))
            type_df = type_df.sample(n=MAX_SUMMARIES, replace=False)
        type_examples = type_df.to_dict('records')
        n = len(type_examples)
        print('Processing {} examples for {} set'.format(n, type))
        if args.single_proc:
            x = list(tqdm(map(generate_samples, type_examples), total=n))
        else:
            x = list(p_uimap(generate_samples, type_examples,  num_cpus=0.8))

        single_extraction_examples = [a[0] for a in x]
        rouge_diffs = [a[1] for a in x]
        rouge_gains = [a[2] for a in x]
        rouge_fulls = [a[3] for a in x]
        output = list(itertools.chain(*single_extraction_examples))
        out_n = len(output)
        account_n = len(set([x['account'] for x in output]))
        out_fn = os.path.join(out_dir, 'single_extraction_labels_{}{}{}.json'.format(type, eval_str, mini_str))
        print('Saving {} labeled single step extraction samples for {} visits to {}'.format(out_n, account_n, out_fn))
        with open(out_fn, 'w') as fd:
            json.dump(output, fd)

        all_rouge_diffs = defaultdict(list)
        all_rouge_gains = defaultdict(list)
        all_rouge_fulls = defaultdict(list)

        for i in range(len(rouge_diffs)):
            for k in rouge_diffs[i]:
                all_rouge_diffs[k].append(rouge_diffs[i][k])
                all_rouge_gains[k].append(rouge_gains[i][k])
                all_rouge_fulls[k].append(rouge_fulls[i][k])

        output_df = []
        for n in all_rouge_diffs:
            v = all_rouge_diffs[n]
            row = {
                'n': n,
                'avg_diff': np.mean(all_rouge_diffs[n]),
                'avg_gain': np.mean(all_rouge_gains[n]),
                'avg_fulls': np.mean(all_rouge_fulls[n]),
                'support': len(all_rouge_diffs[n])
            }
            output_df.append(row)

        output_df = pd.DataFrame(output_df)
        out_fn = 'rouge_stats_{}{}{}.csv'.format(type, eval_str, mini_str)
        print('Saving ROUGE stats by extractive step to {}'.format(out_fn))
        output_df.to_csv(out_fn, index=False)
