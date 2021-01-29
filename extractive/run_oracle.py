from collections import Counter, defaultdict
from functools import partial
import itertools
import json
import math
import os
from string import punctuation
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import numpy as np
import pandas as pd
from p_tqdm import p_uimap

import argparse
from preprocess.constants import out_dir
from preprocess.section_utils import resolve_course, sents_from_html, sent_toks_from_html
from preprocess.utils import get_mrn_status_df, get_records
from evaluations.rouge import max_rouge_sent, max_rouge_set, prepare_str_for_rouge, top_rouge_sents


_TARGET_TOK_CT = 262


def stringify_list(a):
    return ','.join([str(x) for x in a])


def greedy_rel_rouge_recall(source_sents, target_sents):
    return greedy_rel_rouge(source_sents, target_sents, target_tok_ct=recall_target_n)


def greedy_rel_rouge(source_sents, target_sents, target_tok_ct=_TARGET_TOK_CT):
    target = ' '.join(target_sents)
    source_sents_dedup = list(dict.fromkeys(source_sents))
    sent_order, rouge_scores = max_rouge_set(target, source_sents_dedup, rouge_types, target_tok_ct=target_tok_ct)
    summary_sents = []
    for sent_idx in sent_order:
        summary_sents.append(source_sents_dedup[sent_idx])

    n = len(summary_sents)
    summary = ' <s> '.join(summary_sents).strip()
    sum_len = len(summary.split(' ')) - n  # subtract pseudo sentence tokens

    return {
        'sent_order': stringify_list(sent_order),
        'rouge_scores': stringify_list(rouge_scores),
        'prediction': summary,
        'sum_len': sum_len
    }


def random_recall(source_sents, target_sents):
    return random(source_sents, target_sents, target_tok_ct=650)


def random(source_sents, target_sents, target_tok_ct=262):
    source_sents_dedup = list(dict.fromkeys(source_sents))
    sent_order = np.arange(len(source_sents_dedup))
    np.random.shuffle(sent_order)
    summary_sents = []
    sum_len = 0
    for sent_idx in sent_order:
        sum_len += len(source_sents_dedup[sent_idx].split(' '))
        if sum_len > target_tok_ct:
            break
        summary_sents.append(source_sents_dedup[sent_idx])

    n = len(summary_sents)
    summary = ' <s> '.join(summary_sents).strip()
    sum_len = len(summary.split(' ')) - n  # subtract pseudo sentence tokens

    return {
        'prediction': summary,
        'sum_len': sum_len
    }


def top_k_rouge_recall(source_sents, target_sents):
    return top_k_rouge(source_sents, target_sents, target_tok_ct=recall_target_n)


def top_k_rouge(source_sents, target_sents, target_tok_ct=_TARGET_TOK_CT):
    target = ' '.join(target_sents)
    source_sents_dedup = list(dict.fromkeys(source_sents))
    sent_order, rouge_scores = top_rouge_sents(target, source_sents_dedup, rouge_types)
    summary_sents = []
    sent_order_trunc, rouge_scores_trunc = [], []
    sum_len = 0
    for i, sent_idx in enumerate(sent_order):
        sum_len += len(source_sents_dedup[sent_idx].split(' '))
        if sum_len > target_tok_ct:
            break
        summary_sents.append(source_sents_dedup[sent_idx])
        sent_order_trunc.append(sent_idx)
        rouge_scores_trunc.append(rouge_scores[i])

    n = len(summary_sents)
    summary = ' <s> '.join(summary_sents).strip()
    sum_len = len(summary.split(' ')) - n  # subtract pseudo sentence tokens

    return {
        'sent_order': stringify_list(sent_order_trunc),
        'rouge_scores': stringify_list(rouge_scores_trunc),
        'prediction': summary,
        'sum_len': sum_len
    }


def sent_align(source_sents, target_sents):
    n = len(target_sents)
    summary_sents = []
    rouge_scores = []

    source_sents_no_stop = list(map(prepare_str_for_rouge, source_sents))
    for target_sent in target_sents:
        target_sent_no_stop = prepare_str_for_rouge(target_sent)
        _, idx, rouge_score = max_rouge_sent(
            target_sent_no_stop, source_sents_no_stop, rouge_types, return_score=True)
        closest_sent = source_sents[idx]
        if rouge_score > 0.0:
            summary_sents.append(closest_sent)
            rouge_scores.append(rouge_score)

    summary = ' <s> '.join(summary_sents).strip()
    sum_len = len(summary.split(' ')) - n  # subtract pseudo sentence tokens
    return {
        'rouge_scores': stringify_list(rouge_scores),
        'prediction': summary,
        'sum_len': sum_len
    }


def gen_summaries(record):
    target_sents = sents_from_html(resolve_course(record['spacy_target_toks']), convert_lower=True)
    source_sents = sents_from_html(record['spacy_source_toks'], convert_lower=True)
    pred_obj = summarizer(source_sents, target_sents)
    n = len(target_sents)
    reference = ' <s> '.join(target_sents).strip()
    ref_len = len(reference.split(' ')) - n  # subtract pseudo sentence tokens
    obj = {
        'account': record['account'],
        'mrn': record['mrn'],
        'reference': reference,
        'ref_len': ref_len,
    }
    obj.update(pred_obj)
    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate oracle predictions')
    parser.add_argument('--max_n', default=-1, type=int)
    parser.add_argument('--strategy', default='sent_align', choices=
    ['random', 'sent_align', 'greedy_rel', 'greedy_rel_recall', 'top_k', 'top_k_recall', 'random_recall'])
    parser.add_argument('--rouge_types', default='rouge1,rouge2')
    parser.add_argument('--recall_target_n', type=int, default=650)
    parser.add_argument('--custom_path', default=None)
    parser.add_argument('--custom_path_alias', default=None)

    args = parser.parse_args()
    rouge_types = args.rouge_types.split(',')
    recall_target_n = args.recall_target_n

    if args.custom_path is not None:
        records = pd.read_csv(os.path.join(out_dir, args.custom_path)).to_dict('records')
    else:
        mini = 0 <= args.max_n <= 100
        validation_df = get_records(split='validation', mini=mini)
        records = validation_df.to_dict('records')

    if args.strategy == 'sent_align':
        summarizer = sent_align
    elif args.strategy == 'greedy_rel':
        summarizer = greedy_rel_rouge
    elif args.strategy == 'greedy_rel_recall':
        summarizer = greedy_rel_rouge_recall
    elif args.strategy == 'top_k':
        summarizer = top_k_rouge
    elif args.strategy == 'top_k_recall':
        summarizer = top_k_rouge_recall
    elif args.strategy == 'random_recall':
        summarizer = random_recall
    elif args.strategy == 'random':
        summarizer = random

    if args.max_n > 0:
        np.random.seed(1992)
        records = np.random.choice(records, size=args.max_n, replace=False)

    outputs = list(filter(None, p_uimap(gen_summaries, records, num_cpus=0.8)))
    n = len(outputs)
    exp_str = 'oracle_{}'.format(args.strategy)
    alias_str = args.custom_path_alias or 'validation'
    if 'recall' in args.strategy:
        exp_str += '_{}'.format(args.recall_target_n)
    out_fn = os.path.join(out_dir, 'predictions', '{}_{}.csv'.format(exp_str, alias_str))
    print('Saving {} predictions to {}'.format(n, out_fn))
    print('To evaluate, run: cd ../evaluations && python rouge.py --experiment {}'.format(exp_str))
    output_df = pd.DataFrame(outputs)
    output_df.to_csv(out_fn, index=False)
