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
import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluations.rouge import max_rouge_sent
from preprocess.constants import out_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate hybrid oracle retrieval-extractive predictions')
    parser.add_argument('--criteria', default='rouge1', choices=['rouge1', 'rouge2', 'rougeL'])

    args = parser.parse_args()

    retrieval_fn = os.path.join(out_dir, 'predictions', 'retrieval_validation.csv')
    oracle_fn = os.path.join(out_dir, 'predictions', 'oracle_validation.csv')

    retrieval_df = pd.read_csv(retrieval_fn)
    oracle_df = pd.read_csv(oracle_fn)

    oracle_df.sort_values(by=['mrn', 'account'], inplace=True)
    retrieval_df.sort_values(by=['mrn', 'account'], inplace=True)

    oracle_records, retrieval_records = oracle_df.to_dict('records'), retrieval_df.to_dict('records')
    n = len(oracle_records)

    outputs = []
    for i in tqdm(range(n)):
        output = oracle_records[i]
        or_pred = oracle_records[i]['prediction']
        ret_pred = retrieval_records[i]['prediction']

        or_sents = or_pred.split('<s> ')
        ret_sents = ret_pred.split('<s> ')

        ref = oracle_records[i]['reference']
        assert ref == retrieval_records[i]['reference']
        ref_sents = ref.split('<s> ')
        max_sents = []
        for or_sent, ret_sent, ref_sent in zip(or_sents, ret_sents, ref_sents):
            sent = max_rouge_sent(ref_sent.strip(), [or_sent.strip(), ret_sent.strip()], rouge_type=args.criteria)
            max_sents.append(sent)
        n = len(ref_sents)
        summary = ' <s> '.join(max_sents).strip()
        sum_len = len(summary.split(' ')) - n  # subtract pseudo sentence tokens

        output['prediction'] = summary
        output['sum_len'] = sum_len
        outputs.append(output)

    out_fn = os.path.join(out_dir, 'predictions', 'oracle_retrieval_validation.csv')
    output_df = pd.DataFrame(outputs)
    output_df.to_csv(out_fn, index=False)
