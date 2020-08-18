from collections import defaultdict
import os
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from time import sleep

import argparse
from nlp import load_metric
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
from rouge_score import rouge_scorer

from preprocess.constants import out_dir
from utils import decode_utf8


STOPWORDS = set(stopwords.words('english'))


def compute(predictions, references, rouge_types=None, use_agregator=True):
    if rouge_types is None:
        rouge_types = ['rouge1']

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False)
    if use_agregator:
        aggregator = rouge_scorer.scoring.BootstrapAggregator()
    else:
        scores = []

    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        if use_agregator:
            aggregator.add_scores(score)
        else:
            scores.append(score)

    if use_agregator:
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


def max_rouge_set(target, source_sents, rouge_type='rouge2'):
    n = len(source_sents)
    target_no_stop = prepare_str_for_rouge(target)
    source_sents_no_stop = list(map(prepare_str_for_rouge, source_sents))
    curr_sum = ''
    curr_rouge = 0.0
    sent_order = []
    rouges = []
    for _ in range(n):
        _, idx, score = max_rouge_sent(
            target_no_stop, source_sents_no_stop, rouge_type=rouge_type, return_score=True, source_prefix=curr_sum,
            mask_idxs=sent_order
        )
        if score <= curr_rouge:
            break
        curr_rouge = score
        curr_sum += source_sents[idx] + ' '
        sent_order.append(idx)
        rouges.append(curr_rouge)
    return sent_order, rouges


def max_rouge_sent(target, source_sents, rouge_type='rougeL', return_score=False, source_prefix='', mask_idxs=[]):
    n = len(source_sents)
    predictions = [source_prefix + s for s in source_sents]
    references = [target for _ in range(n)]
    outputs = compute(
        predictions=predictions, references=references, rouge_types=[rouge_type], use_agregator=False)
    scores = np.array(list(map(lambda x: x.fmeasure, outputs[rouge_type])))
    if len(mask_idxs) > 0:
        scores[mask_idxs] = float('-inf')
    max_idx = np.argmax(scores)
    max_source_sent = source_sents[max_idx]
    if return_score:
        return max_source_sent, max_idx, scores[max_idx]
    return max_source_sent, max_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ROUGE Scorer')
    parser.add_argument('--experiment', default='lr')

    args = parser.parse_args()

    metric = load_metric('rouge')

    in_fn = os.path.join(out_dir, 'predictions', '{}_validation.csv'.format(args.experiment))
    print('Loading predictions from {}...'.format(in_fn))
    df = pd.read_csv(in_fn)
    print('Loaded {} predictions.'.format(df.shape[0]))
    predictions = df['prediction'].tolist()
    references = df['reference'].tolist()
    rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    print('Computing {}...'.format(', '.join(rouge_types)))
    scores = metric.compute(predictions=predictions, references=references, rouge_types=rouge_types)

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

    out_fn = os.path.join('results', args.experiment)
    print('Saving results to {}'.format(out_fn))
    results_df.to_csv(out_fn, index=False)