import itertools
import json
import pickle
import os
import sys

import numpy as np
from torch.utils.data import Dataset

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from egrids.vocab import Vocab
from preprocess.constants import out_dir


class EGridDataset(Dataset):
    def __init__(self, vocab, split, k=3):
        in_fn = os.path.join(out_dir, 'egrids.json')
        print('Loading {} set from {}'.format(split, in_fn))
        with open(in_fn, 'r') as fd:
            all_examples = json.load(fd)
        print('Done Loading...')
        self.examples = [ex for ex in all_examples if ex['split'] == split and len(ex['egrid']) > 0]
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
    with open('data/vocab.pk', 'rb') as fd:
        vocab = pickle.load(fd)

    dataset = EGrid(vocab, 'validation', k=3)
    print(dataset[0])
