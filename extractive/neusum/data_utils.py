import itertools
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.generate_extractive_mmr_samples import Example
from preprocess.constants import out_dir
from extractive.neusum.vocab import Vocab

MAX_SENT_LEN = 30
MAX_CURR_SUM_SENTS = 30
MAX_SOURCE_SENTS = 100
LABEL_TEMP = 10.0


def min_max_norm(arr):
    s, e = arr.min(), arr.max()
    denom = max(e - s, 1e-5)
    return (arr - s) / denom


def softmax(arr, temp=1.0):
    arr_exp = np.exp(arr * temp)
    return arr_exp / arr_exp.sum()


def collate_fn(batch):
    """
    :param batch:
    :return:
    """
    source_ids = [b['source_ids'] for b in batch]  # batch_size, num_sent, max_sents
    sum_ids = [b['sum_ids'] for b in batch]

    target_dist = [torch.FloatTensor(b['target_dist']) for b in batch]

    source_lens = [len(sid) for sid in source_ids]
    source_sent_lens = [[len(x) for x in sid] for sid in source_ids]
    source_ids_flat = list(map(torch.LongTensor, list(itertools.chain(*(source_ids)))))
    source_ids_flat_pad = pad_sequence(source_ids_flat, batch_first=True, padding_value=0)
    source_sent_lens_flat = list(itertools.chain(*(source_sent_lens)))

    sum_lens = [len(sid) for sid in sum_ids]
    sum_sent_lens = [[len(x) for x in sid] for sid in sum_ids]
    sum_ids_flat = list(map(torch.LongTensor, list(itertools.chain(*(sum_ids)))))
    sum_ids_flat_pad = pad_sequence(sum_ids_flat, batch_first=True, padding_value=0)
    sum_sent_lens_flat = list(itertools.chain(*(sum_sent_lens)))

    target_dist_pad = pad_sequence(target_dist, batch_first=True, padding_value=0.0)

    counts = {
        'source_sent_lens_flat': torch.LongTensor(source_sent_lens_flat),
        'source_lens': torch.LongTensor(source_lens),
        'sum_sent_lens_flat': torch.LongTensor(sum_sent_lens_flat),
        'sum_lens': torch.LongTensor(sum_lens),
    }

    return source_ids_flat_pad, sum_ids_flat_pad, target_dist_pad, counts


def truncate(str, max_len):
    arr = str.split(' ')
    end_idx = min(len(arr), max_len)
    return arr[:end_idx]


class SingleExtractionDataset(Dataset):
    def __init__(self, vocab):
        # in_fn = os.path.join(out_dir, 'single_extraction_labels.pk')
        in_fn = 'data/single_extraction_labels.pk'
        with open(in_fn, 'rb') as fd:
            self.examples = pickle.load(fd)
        self.vocab = vocab
        self.vocab.add_token('<s>')

    def str_to_ids(self, str):
        return self.to_ids(str.split(' '))

    def to_ids(self, arr):
        return self.vocab.get_ids(arr)

    def __getitem__(self, item):
        example = self.examples[item]
        source_sents = example.candidate_source_sents
        rouges = example.target_rouges
        seen = set()
        source_sents_dedup = []
        rouges_dedup = []
        for rouge, source_sent in zip(rouges, source_sents):
            if source_sent in seen:
                continue
            else:
                source_sents_dedup.append(source_sent)
                rouges_dedup.append(rouge)
                seen.add(source_sent)

        source_sents_dedup = [truncate(x, MAX_SENT_LEN) for x in source_sents_dedup]
        if len(example.curr_sum_sents) == 0:
            curr_sum_sents = [['<s>']]
        else:
            curr_sum_sents = [truncate(x, MAX_SENT_LEN) for x in example.curr_sum_sents]

        rel_rouges = np.array(rouges_dedup) - example.curr_rouge
        curr_sum_sents = curr_sum_sents[:min(len(curr_sum_sents), MAX_CURR_SUM_SENTS)]
        n = len(source_sents_dedup)
        if n > MAX_SOURCE_SENTS:
            np.random.seed(1992)
            sample_idxs = np.random.choice(range(n), size=MAX_SOURCE_SENTS, p=softmax(rel_rouges), replace=False)
            rel_rouges = np.array([rel_rouges[idx] for idx in sample_idxs])
            source_sents_dedup = [source_sents_dedup[idx] for idx in sample_idxs]
        target_dist = softmax(min_max_norm(rel_rouges), temp=LABEL_TEMP)
        sum_ids = list(map(self.to_ids, curr_sum_sents))
        source_ids = list(map(self.to_ids, source_sents_dedup))
        return {
            'sum_ids': sum_ids,
            'source_ids': source_ids,
            'target_dist': target_dist,
            'rel_rouges': rel_rouges,
        }

    def __len__(self):
        return len(self.examples)


if __name__ == '__main__':
    # print('Loading vocabulary...')
    # with open('data/vocab_num_template.pk', 'rb') as fd:
    #     vocab = pickle.load(fd)
    # print('Constructing dataset')
    # dataset = SingleExtractionDataset(vocab)
    # print(dataset[0])

    in_fn = os.path.join(out_dir, 'single_extraction_labels.pk')
    with open(in_fn, 'rb') as fd:
        examples = pickle.load(fd)

    print([x.curr_sum_sents for x in examples])