import itertools
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import out_dir
from extractive.neusum.vocab import Vocab

_DEFAULT_LABEL_TEMP = 10.0
MAX_SENT_LEN = 30
MAX_CURR_SUM_SENTS = 30
MAX_SOURCE_SENTS = 100


def mask_2D(batch_size, max_seq_len, seq_lens):
    mask = torch.BoolTensor(size=(batch_size, max_seq_len))
    mask.fill_(False)
    for batch_idx, seq_len in enumerate(seq_lens):
        if seq_len < max_seq_len:
            mask[batch_idx, seq_len:] = True
    return mask


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
    batch_size = len(batch)

    source_ids = [b['source_ids'] for b in batch]  # batch_size, num_sent, max_sents
    sum_ids = [b['sum_ids'] for b in batch]

    target_dist = [torch.FloatTensor(b['target_dist']) for b in batch]

    source_lens = [len(sid) for sid in source_ids]
    source_sent_lens = [[len(x) for x in sid] for sid in source_ids]
    source_ids_flat = list(map(torch.LongTensor, list(itertools.chain(*(source_ids)))))
    source_ids_flat_pad = pad_sequence(source_ids_flat, batch_first=True, padding_value=0)
    source_sent_lens_flat = list(itertools.chain(*(source_sent_lens)))
    _, max_source_sents = source_ids_flat_pad.size()
    source_mask = mask_2D(batch_size, max(source_lens), source_lens)

    sum_lens = [len(sid) for sid in sum_ids]
    sum_sent_lens = [[len(x) for x in sid] for sid in sum_ids]
    sum_ids_flat = list(map(torch.LongTensor, list(itertools.chain(*(sum_ids)))))
    sum_ids_flat_pad = pad_sequence(sum_ids_flat, batch_first=True, padding_value=0)
    sum_sent_lens_flat = list(itertools.chain(*(sum_sent_lens)))
    _, max_sum_len = sum_ids_flat_pad.size()
    sum_att_mask = mask_2D(batch_size, max_sum_len, sum_lens)

    target_dist_pad = pad_sequence(target_dist, batch_first=True, padding_value=0.0)

    counts = {
        'source_sent_lens_flat': torch.LongTensor(source_sent_lens_flat),
        'source_lens': torch.LongTensor(source_lens),
        'sum_sent_lens_flat': torch.LongTensor(sum_sent_lens_flat),
        'sum_lens': torch.LongTensor(sum_lens),
    }

    masks = {
        'sum_att_mask': sum_att_mask,
        'source_mask': source_mask,
    }

    return source_ids_flat_pad, sum_ids_flat_pad, target_dist_pad, counts, masks


def truncate(str, max_len):
    arr = str.split(' ')
    end_idx = min(len(arr), max_len)
    return arr[:end_idx]


class SingleExtractionDataset(Dataset):
    def __init__(self, vocab, type, mini, label_temp=_DEFAULT_LABEL_TEMP):
        mini_str = '_mini' if mini else ''
        in_fn = os.path.join(out_dir, 'single_extraction_labels_{}{}.json'.format(type, mini_str))
        print('Loading {} set from {}'.format(type, in_fn))
        with open(in_fn, 'r') as fd:
            self.examples = json.load(fd)
        print('Finished loading {} set'.format(type))
        self.vocab = vocab
        self.label_temp = label_temp

    def str_to_ids(self, str):
        return self.to_ids(str.split(' '))

    def to_ids(self, arr):
        return self.vocab.get_ids(arr)

    def __getitem__(self, item):
        example = self.examples[item]
        source_sents = example['candidate_source_sents']
        rouges = example['target_rouges']
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
        if len(example['curr_sum_sents']) == 0:
            curr_sum_sents = [['<s>']]
        else:
            curr_sum_sents = [truncate(x, MAX_SENT_LEN) for x in example['curr_sum_sents']]

        rel_rouges = np.array(rouges_dedup) - example['curr_rouge']
        curr_sum_sents = curr_sum_sents[:min(len(curr_sum_sents), MAX_CURR_SUM_SENTS)]
        n = len(source_sents_dedup)
        if n > MAX_SOURCE_SENTS:
            np.random.seed(1992)
            sample_idxs = np.random.choice(range(n), size=MAX_SOURCE_SENTS, p=softmax(rel_rouges), replace=False)
            rel_rouges = np.array([rel_rouges[idx] for idx in sample_idxs])
            source_sents_dedup = [source_sents_dedup[idx] for idx in sample_idxs]
        target_dist = softmax(min_max_norm(rel_rouges), temp=self.label_temp)
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
