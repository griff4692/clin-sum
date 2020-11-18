from collections import defaultdict
import itertools
import os
import random
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import numpy as np
import pandas as pd
from p_tqdm import p_uimap
from tqdm import tqdm
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

import argparse
from preprocess.constants import out_dir
from preprocess.section_utils import resolve_course, sents_from_html, sent_toks_from_html
from preprocess.utils import get_records


MAX_PER_EXAMPLE = 25


def nsp(s1, s2):
    encoding = tokenizer(s1, s2, return_tensors='pt')
    outputs = model(**encoding, next_sentence_label=torch.LongTensor([1]))
    logits = outputs.logits
    prob = torch.softmax(logits, dim=1)
    return float(prob[0, 0])


def process(target_sents):
    vals = defaultdict(list)
    num_sents = len(target_sents)
    combs = list(itertools.combinations(range(num_sents), 2))
    if len(combs) > MAX_PER_EXAMPLE:
        np.random.shuffle(combs)
        combs = combs[:MAX_PER_EXAMPLE]
    for i, j in combs:
        small_idx = min(i, j)
        large_idx = max(i, j)
        assert small_idx < large_idx
        delta = large_idx - small_idx
        vals[delta].append(nsp(target_sents[small_idx], target_sents[large_idx]))
        vals[-delta].append(nsp(target_sents[large_idx], target_sents[small_idx]))
    return vals


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate NSP scores')
    parser.add_argument('--pretrained_model', default='emilyalsentzer/Bio_ClinicalBERT', choices=['emilyalsentzer/Bio_ClinicalBERT'])
    parser.add_argument('--max_n', default=-1, type=int)

    args = parser.parse_args()

    mini = 0 <= args.max_n <= 100
    validation_df = get_records(split='validation', mini=mini)
    records = validation_df.to_dict('records')
    if args.max_n > 0:
        np.random.seed(1992)
        records = np.random.choice(records, size=args.max_n, replace=False)
    target_sents = [sents_from_html(record['spacy_target_toks']) for record in records]

    n = len(records)
    print('Loading tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    print('Loading model...')
    model = BertForNextSentencePrediction.from_pretrained(args.pretrained_model, return_dict=True)

    print('Generating NSP predictions for {} examples'.format(n))
    outputs = list(tqdm(map(process, target_sents), total=n))
    agg_output = defaultdict(list)
    for output in outputs:
        for k, v in output.items():
            agg_output[k] += v
    all_keys = list(sorted(agg_output.keys()))
    output_df = {'offset': [], 'nsp': [], 'support': []}
    for k in all_keys:
        v = agg_output[k]
        mean = np.mean(v)
        support = len(v)
        print(k, mean, len(v))
        output_df['offset'].append(k)
        output_df['nsp'].append(mean)
        output_df['support'].append(len(v))

    output_df = pd.DataFrame(output_df)
    out_fn = 'results/nsp/clinsum.csv'
    output_df.to_csv(out_fn, index=False)
