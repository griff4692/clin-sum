from collections import defaultdict
import json
import itertools
import os
import random
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import argparse
from datasets import load_dataset
import numpy as np
import pandas as pd
from p_tqdm import p_uimap
import spacy
from tqdm import tqdm
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch


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
    parser = argparse.ArgumentParser('Script to generate NSP scores for other datasets')
    parser.add_argument('--dataset', default='cnn_dm', choices=[
        'cnn_dm', 'newsroom', 'arxiv', 'pubmed'
    ])
    parser.add_argument('--max_n', default=1000, type=int)

    args = parser.parse_args()
    # spacy_nlp = spacy.load('en_core_sci_lg', disable=['tagger', 'parser', 'ner', 'textcat'])
    # spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))
    if args.dataset == 'cnn_dm':
        data = load_dataset('cnn_dailymail', '3.0.0', split='validation[:{}]'.format(args.max_n))
        model_name = 'bert-base-cased'
        target_sents = [text.split('\n') for text in data['highlights']]
    elif args.dataset == 'newsroom':
        data = load_dataset('newsroom', split='validation[:{}]'.format(args.max_n))
        target_sents = []
        model_name = None
    elif args.dataset == 'pubmed':
        model_name = 'monologg/biobert_v1.1_pubmed'
        with open('../data/datasets/pubmed/val.txt', 'r') as fd:
            data = [x for x in fd.readlines() if len(x) > 0]
        np.random.shuffle(data)
        data = data[:args.max_n]
        data = list(map(json.loads, data))
        target_sents = [[y[3:-4].strip() for y in x['abstract_text']] for x in data]
    elif args.dataset == 'arxiv':
        model_name = 'bert-base-cased'
        with open('../data/datasets/arxiv/val.txt', 'r') as fd:
            data = [x for x in fd.readlines() if len(x) > 0]
        np.random.shuffle(data)
        data = data[:args.max_n]
        data = list(map(json.loads, data))
        target_sents = [[y[3:-4].strip() for y in x['abstract_text']] for x in data]
    else:
        data = None
        model_name = None
        target_sents = None
    n = len(data)
    print('Loading tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print('Loading model...')
    model = BertForNextSentencePrediction.from_pretrained(model_name, return_dict=True)
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
    out_fn = 'results/nsp/{}.csv'.format(args.dataset)
    output_df.to_csv(out_fn, index=False)
