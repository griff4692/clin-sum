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

Example = namedtuple(
    'Example',
    ['mrn', 'account', 'curr_sum_sents', 'candidate_source_sents', 'curr_rouge', 'target_rouges', 'target_sents']
)


def extraction_is_keep(str):
    MIN_WORD_CT = 3
    any_alpha = re.search('[a-zA-Z]', str) is not None
    toks = str.split(' ')
    long_enough = len(toks) >= MIN_WORD_CT
    return any_alpha and long_enough


def generate_samples(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    examples_fn = os.path.join(mrn_dir, 'examples.csv')
    examples_df = pd.read_csv(examples_fn)
    examples_df.dropna(inplace=True)
    assert len(examples_df) > 0

    single_extraction_examples = []

    for row in examples_df.to_dict('records'):
        source_sents = sents_from_html(row['spacy_source_toks'], convert_lower=True)
        target_sents = sents_from_html(row['spacy_target_toks'], convert_lower=True)
        target = ' '.join(target_sents)
        target_no_stop = prepare_str_for_rouge(target)

        source_sents_no_stop = list(map(prepare_str_for_rouge, source_sents))
        # remove 1-2 word sentences (too many of them) and most are not necessary for BHC
        keep_idxs = [idx for idx, s in enumerate(source_sents_no_stop) if extraction_is_keep(s)]
        source_sents_no_stop_filt = [source_sents_no_stop[idx] for idx in keep_idxs]
        source_n = len(keep_idxs)
        curr_sum_sents = []
        curr_rouge = 0.0
        included_sent_idxs = set()
        max_target_n = min(source_n, len(target_sents))
        references = [target_no_stop for _ in range(source_n)]
        for _ in range(max_target_n):
            curr_sum = ' '.join(curr_sum_sents).strip() + ' '
            predictions = [(curr_sum + s).strip() for s in source_sents_no_stop_filt]
            outputs = compute(
                predictions=predictions, references=references, rouge_types=['rouge1'], use_agregator=False)
            scores = np.array(list(map(lambda x: x.fmeasure, outputs['rouge1'])))
            if len(included_sent_idxs) > 0:
                scores[list(included_sent_idxs)] = float('-inf')
            max_idx = np.argmax(scores)
            score = scores[max_idx]
            if score <= curr_rouge:
                break

            eligible_scores = []
            eligible_source_sents = []
            for i in range(len(scores)):
                if i not in included_sent_idxs:
                    eligible_scores.append(scores[i])
                    eligible_source_sents.append(source_sents_no_stop_filt[i])
            # Example
            example = Example(
                mrn=mrn,
                account=row['account'],
                curr_sum_sents=curr_sum_sents.copy(),
                candidate_source_sents=eligible_source_sents,
                curr_rouge=curr_rouge,
                target_rouges=eligible_scores,
                target_sents=target_sents
            )
            single_extraction_examples.append(example)
            curr_rouge = score
            curr_sum_sents.append(source_sents_no_stop_filt[int(max_idx)])
            included_sent_idxs.add(max_idx)

    return single_extraction_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate greedy labels for single-step extraction task.')
    parser.add_argument('--max_n', type=int, default=-1)

    args = parser.parse_args()

    _, _, mrns = get_mrn_status_df('valid_example')
    if args.max_n > 0:
        size = min(args.max_n, len(mrns))
        np.random.seed(1992)
        mrns = np.random.choice(mrns, size=size, replace=False)
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    if True:  # args.max_n < 100:
        single_extraction_examples = list(tqdm(map(generate_samples, mrns), total=len(mrns)))
    else:
        single_extraction_examples = list(p_uimap(generate_samples, mrns))
    single_extraction_examples_flat = list(itertools.chain(*single_extraction_examples))
    out_fn = os.path.join(out_dir, 'single_extraction_labels.pk')
    with open(out_fn, 'wb') as fd:
        pickle.dump(single_extraction_examples_flat, fd)
