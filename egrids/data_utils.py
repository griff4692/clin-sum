import itertools
import json
import pickle
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from egrids.vocab import Vocab
from preprocess.constants import out_dir


class EGridDataset(Dataset):
    def __init__(self, vocab, split, k=3, mini=False, egrids=None):
        if egrids is None:
            if mini:
                in_fn = os.path.join(out_dir, 'egrids_small.json')
            else:
                in_fn = os.path.join(out_dir, 'egrids.json')
            print('Loading {} set from {}'.format(split, in_fn))
            with open(in_fn, 'r') as fd:
                all_examples = json.load(fd)
            print('Done Loading...')
            self.examples = [ex for ex in all_examples if ex['split'] == split]
        else:
            self.examples = egrids
        self.examples = [ex for ex in self.examples if len(ex['egrid']) > 0]
        self.vocab = vocab
        self.k = k

    def get_features(self, egrid, num_sents, sent_order):
        local_ids = []
        pad_id = self.vocab.get_id(Vocab.PAD_TOKEN)
        null_id = self.vocab.get_id(Vocab.NULL_TOKEN)
        for cui in egrid:
            cui_id = self.vocab.get_id(cui)
            trans = [null_id] * num_sents
            for location in egrid[cui]:
                trans[sent_order[location['sent_idx']]] = cui_id
            trans = ([pad_id] * (self.k - 1)) + trans + ([pad_id] * (self.k - 1))
            full_len = len(trans)
            for i in range(0, full_len - self.k + 1):
                local_ids.append(tuple(trans[i:i + self.k]))
        return [list(x) for x in list(set(local_ids))]

    def __getitem__(self, item):
        """
        :param item: example index
        :return: local ids for all transitions of variable length
        """
        example = self.examples[item]
        egrid = example['egrid']
        num_sents = example['num_target_sents']
        forward_sent_order = np.arange(num_sents)
        rand_sent_order = np.arange(num_sents)
        np.random.shuffle(rand_sent_order)

        positive_ids = self.get_features(egrid, num_sents, forward_sent_order)
        negative_ids = self.get_features(egrid, num_sents, rand_sent_order)
        return {'positive_ids': positive_ids, 'negative_ids': negative_ids}

    def __len__(self):
        return len(self.examples)


if __name__ == '__main__':
    sparsity_df = []
    in_fn = os.path.join(out_dir, 'egrids.json')
    with open(in_fn, 'r') as fd:
        all_examples = json.load(fd)

    n = len(all_examples)
    print('Processing...')
    for ex_idx in tqdm(range(n)):
        example = all_examples[ex_idx]
        egrid = example['egrid']
        cui_info = example['cui_info']
        for cui in egrid:
            tui = cui_info[cui]['tui']
            sem_group = cui_info[cui]['sem_group']
            locations = egrid[cui]
            num_mentions = len(locations)
            sent_idxs = list(sorted([l['sent_idx'] for l in locations]))
            num_adjacent = 0
            num_same = 0
            for i in range(1, num_mentions):
                if sent_idxs[i - 1] == sent_idxs[i]:
                    num_same += 1
                if sent_idxs[i - 1] + 1 == sent_idxs[i]:
                    num_adjacent += 1
            sparsity_df.append({
                'cui': cui,
                'tui': tui,
                'sem_group': sem_group,
                'num_mentions': num_mentions,
                'num_adjacent': num_adjacent,
                'num_same': num_same
            })

    out_fn = 'stats/entity_distribution.csv'
    df = pd.DataFrame(sparsity_df)
    print('Saving {} rows to {}'.format(len(df), out_fn))
    df.to_csv(out_fn, index=False)

    n = len(df)
    single_mentions_df = df[df['num_mentions'] == 1]
    multiple_mentions_df = df[df['num_mentions'] > 1]
    mult_n = len(multiple_mentions_df)
    print('Number of rows={}. Number of single mentions={}'.format(n, len(single_mentions_df)))

    denom = float(multiple_mentions_df['num_mentions'].sum() - mult_n)
    adj_freq = multiple_mentions_df['num_adjacent'].sum() / denom
    same_freq = multiple_mentions_df['num_same'].sum() / denom

    print('Adjacent likelihood={}. Same likelihood={}'.format(adj_freq, same_freq))
