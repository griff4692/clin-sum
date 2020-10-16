import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from evaluations.rouge import compute, prepare_str_for_rouge


if __name__ == '__main__':
    in_fn = os.path.join(out_dir, 'predictions', 'oracle_greedy_rel_hiv_validation.csv')
    print('Loading predictions from {}...'.format(in_fn))
    df = pd.read_csv(in_fn)
    df['prediction'] = df['prediction'].fillna(value='')

    print('Loaded {} predictions.'.format(df.shape[0]))
    predictions = list(map(prepare_str_for_rouge, df['prediction'].tolist()))
    references = list(map(prepare_str_for_rouge, df['reference'].tolist()))
    n = len(df)
    rouge_types = ['rouge1', 'rouge2']
    outputs = compute(predictions, references, rouge_types=rouge_types, use_aggregator=False)
    f1_means = np.array(
        [sum([outputs[t][i].fmeasure for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    recall_means = np.array(
        [sum([outputs[t][i].recall for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    precision_means = np.array(
        [sum([outputs[t][i].precision for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])

    df['f1'] = f1_means
    df['recall'] = recall_means
    df['precision'] = precision_means

    out_fn = os.path.join(out_dir, 'human', 'oracle_greedy_rel_hiv_validation_scored.csv')
    print('Saving {} ROUGE-scored examples to {}'.format(len(df), out_fn))
    df.to_csv(out_fn, index=False)
