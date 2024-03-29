from collections import defaultdict
import os
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))
import string
from time import sleep

import argparse
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
from p_tqdm import p_imap
from rouge_score import rouge_scorer
from tqdm import tqdm

from preprocess.constants import out_dir
from utils import decode_utf8

PATIENT_TERMS = {'patient', 'pt', 'patient\'s', 'patients', 'patients\''}
STOPWORDS = set(stopwords.words('english')).union(string.punctuation).union(PATIENT_TERMS)


def compute(predictions, references, rouge_types=None, use_aggregator=True, use_parallel=False, show_progress=False):
    if rouge_types is None:
        rouge_types = ['rouge1']

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False)
    aggregator = rouge_scorer.scoring.BootstrapAggregator() if use_aggregator else None
    if not use_aggregator and use_parallel:
        scores = list(p_imap(lambda x: scorer.score(x[0], x[1]), list(zip(references, predictions))))
    else:
        scores = []
        if show_progress:
            for i in tqdm(range(len(references))):
                score = scorer.score(references[i], predictions[i])
                if use_aggregator:
                    aggregator.add_scores(score)
                else:
                    scores.append(score)
        else:
            for i in range(len(references)):
                score = scorer.score(references[i], predictions[i])
                if use_aggregator:
                    aggregator.add_scores(score)
                else:
                    scores.append(score)

    if use_aggregator:
        result = aggregator.aggregate()
    else:
        result = {}
        for key in scores[0]:
            result[key] = list(score[key] for score in scores)

    return result


def prepare_str_for_rouge(str):
    return decode_utf8(remove_stopwords(str))


def remove_stopwords(str):
    tok = str.split(' ')
    return ' '.join([t for t in tok if not t in STOPWORDS])


def top_rouge_sents(target, source_sents, rouge_types):
    n = len(source_sents)
    target_no_stop = prepare_str_for_rouge(target)
    source_sents_no_stop = list(map(prepare_str_for_rouge, source_sents))
    references = [target_no_stop for _ in range(n)]
    outputs = compute(
        predictions=source_sents_no_stop, references=references, rouge_types=rouge_types, use_aggregator=False)
    scores = np.array([sum([outputs[t][i].fmeasure for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])

    sent_order = scores.argsort()[::-1]
    rouges = [scores[i] for i in sent_order]

    return sent_order, rouges


def max_rouge_set(target, source_sents, rouge_types, target_tok_ct=None):
    n = len(source_sents)
    target_no_stop = prepare_str_for_rouge(target)
    source_sents_no_stop = list(map(prepare_str_for_rouge, source_sents))
    curr_sum = ''
    curr_rouge = 0.0
    sent_order = []
    rouges = []
    metric = 'f1' if target_tok_ct is None else 'recall'
    for _ in range(n):
        _, idx, score = max_rouge_sent(
            target_no_stop, source_sents_no_stop, rouge_types, return_score=True, source_prefix=curr_sum,
            mask_idxs=sent_order, metric=metric
        )

        decreasing_score = score <= curr_rouge
        mc = target_tok_ct is not None and len(source_sents[idx].split(' ')) + len(curr_sum.split(' ')) > target_tok_ct

        if decreasing_score or mc:
            break
        curr_rouge = score
        curr_sum += source_sents[idx] + ' '
        sent_order.append(idx)
        rouges.append(curr_rouge)
    return sent_order, rouges


def max_rouge_sent(target, source_sents, rouge_types, return_score=False, source_prefix='', mask_idxs=[], metric='f1'):
    n = len(source_sents)
    predictions = [source_prefix + s for s in source_sents]
    references = [target for _ in range(n)]
    outputs = compute(
        predictions=predictions, references=references, rouge_types=rouge_types, use_aggregator=False)
    if metric == 'f1':
        scores = np.array(
            [sum([outputs[t][i].fmeasure for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    elif metric == 'recall':
        scores = np.array(
            [sum([outputs[t][i].recall for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    elif metric == 'precision':
        scores = np.array(
            [sum([outputs[t][i].precision for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    if len(mask_idxs) > 0:
        scores[mask_idxs] = float('-inf')
    max_idx = np.argmax(scores)
    max_source_sent = source_sents[max_idx]
    if return_score:
        return max_source_sent, max_idx, scores[max_idx]
    return max_source_sent, max_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ROUGE Scorer')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('-rougeL', default=False, action='store_true')

    args = parser.parse_args()

    in_fn = os.path.join(out_dir, 'predictions', '{}.csv'.format(args.experiment))
    if not os.path.exists(in_fn):
        in_fn = os.path.join(out_dir, 'predictions', '{}_validation.csv'.format(args.experiment))
    print('Loading predictions from {}...'.format(in_fn))
    df = pd.read_csv(in_fn)
    df['prediction'] = df['prediction'].fillna(value='')

    print('Loaded {} predictions.'.format(df.shape[0]))
    predictions = list(map(prepare_str_for_rouge, df['prediction'].tolist()))
    references = list(map(prepare_str_for_rouge, df['reference'].tolist()))
    rouge_types = ['rouge1', 'rouge2']
    if args.rougeL:  # this is very slow
        rouge_types.append('rougeL')
    print('Computing {}...'.format(', '.join(rouge_types)))
    scores = compute(predictions=predictions, references=references, rouge_types=rouge_types, show_progress=True)

    results = []
    stats = ['low', 'mid', 'high']
    measures = ['precision', 'recall', 'fmeasure']
    print('Showing mean statistics...')
    for name, agg_score in scores.items():
        print(name)
        for stat in stats:
            score_obj = getattr(agg_score, stat)
            for measure in measures:
                value = getattr(score_obj, measure)
                results.append({
                    'name': name,
                    'measure': measure,
                    'agg_type': stat,
                    'value': value
                })
                if stat == 'mid':
                    print('\t{}={}'.format(measure, value))
        print('\n')
    results_df = pd.DataFrame(results)

    out_fn = os.path.join('results', '{}.csv'.format(args.experiment))
    print('Saving results to {}'.format(out_fn))
    results_df.to_csv(out_fn, index=False)
