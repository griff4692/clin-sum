import os
import pickle
import sys

import numpy as np
from torch.utils.data import Dataset

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.generate_extractive_mmr_samples import Example
from preprocess.constants import out_dir
from extractive.neusum.vocab import Vocab

MAX_CURR_SUM_SENTS = 20
MAX_SOURCE_SENTS = 200
LABEL_TEMP = 10.0


def min_max_norm(arr):
    s, e = arr.min(), arr.max()
    denom = max(e - s, 1e-5)
    return (arr - s) / denom


def softmax(arr, temp=1):
    arr_exp = np.exp(arr * temp)
    return arr_exp / arr_exp.sum()


class SingleExtractionDataset(Dataset):
    def __init__(self, vocab, type='train'):
        in_fn = os.path.join(out_dir, 'single_extraction_labels.pk')
        with open(in_fn, 'rb') as fd:
            self.examples = pickle.load(fd)
        self.vocab = vocab

    def str_to_ids(self, str):
        return self.vocab.get_ids(str.split(' '))

    def __getitem__(self, item):
        example = self.examples[item]
        source_sents = example.candidate_source_sents
        rel_rouges = np.array(example.target_rouges) - example.curr_rouge
        curr_sum_sents = example.curr_sum_sents

        curr_sum_sents = curr_sum_sents[:min(len(curr_sum_sents), MAX_CURR_SUM_SENTS)]
        n = len(source_sents)
        if n > MAX_SOURCE_SENTS:
            sample_idxs = np.random.choice(range(n), size=MAX_SOURCE_SENTS, p=softmax(rel_rouges), replace=False)
            rel_rouges = np.array([rel_rouges[idx] for idx in sample_idxs])
            source_sents = [source_sents[idx] for idx in sample_idxs]
        target_dist = softmax(min_max_norm(rel_rouges), temp=LABEL_TEMP)
        sum_ids = list(map(self.str_to_ids, curr_sum_sents))
        source_ids = list(map(self.str_to_ids, source_sents))
        return {
            'sum_ids': sum_ids,
            'source_ids': source_ids,
            'target_dist': target_dist,
            'rel_rouges': rel_rouges
        }

    def __len__(self):
        return len(self.examples)


if __name__ == '__main__':
    print('Loading vocabulary...')
    with open('data/vocab.pk', 'rb') as fd:
        vocab = pickle.load(fd)
    print('Constructing dataset')
    dataset = SingleExtractionDataset(vocab)
    print(dataset[0])