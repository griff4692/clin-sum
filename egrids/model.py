import json
import glob
import os
import multiprocessing
import pickle
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import argparse
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from egrids.data_utils import EGridDataset
from egrids.vocab import Vocab
from preprocess.constants import out_dir
from utils import tens_to_np


class EntityGridModel(pl.LightningModule):
    def __init__(self, args, vocab):
        super(EntityGridModel, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(len(vocab), embedding_dim=args.embedding_dim)

        self.maps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self['k'] * args.embedding_dim, 1),
                nn.ReLU()
            ) for _ in range(self['num_maps'])
        ])

        self.output = nn.Linear(self['num_maps'], 1)

    def on_epoch_start(self):
        print('\n')

    def __getitem__(self, item):
        return getattr(self.args, item)

    def shared_step(self, batch, num_samples):
        num_windows, window_size = batch.size()
        embeds = self.embedding(batch)
        embeds_concat = embeds.view(num_windows, -1)
        fmaps = torch.cat([
            mapper(embeds_concat) for mapper in self.maps
        ], dim=1)

        max_chunks = []
        start_idx = 0
        for num in num_samples:
            max_chunk = torch.max(fmaps[start_idx:start_idx + num], dim=0)[0]
            max_chunks.append(max_chunk.unsqueeze(0))
            start_idx += num
        max_feats = torch.cat(max_chunks, dim=0)
        return self.output(max_feats)

    def training_step(self, batch, batch_idx):
        pos_score = self.shared_step(batch['positive_ids'], batch['positive_samples'])
        neg_score = self.shared_step(batch['negative_ids'], batch['negative_samples'])
        loss = torch.clamp(1 - pos_score + neg_score, min=0).mean()
        return loss

    def validation_step(self, batch, batch_idx):
        pos_score = self.shared_step(batch['positive_ids'], batch['positive_samples'])
        neg_score = self.shared_step(batch['negative_ids'], batch['negative_samples'])
        loss = torch.clamp(1 - pos_score + neg_score, min=0).mean()

        return loss

    def test_step(self, batch, batch_idx):
        pos_score = self.shared_step(batch['positive_ids'], batch['positive_samples'])
        neg_score = self.shared_step(batch['negative_ids'], batch['negative_samples'])
        return neg_score, pos_score

    def test_epoch_end(self, outputs) -> None:
        # this out is now the full size of the batch
        num_correct = 0
        num_tested = 0
        for output in outputs:
            neg_score, pos_score = output
            batch_size = len(neg_score)
            num_correct += (pos_score > neg_score).sum()
            num_tested += batch_size
        accuracy = float(num_correct) / float(num_tested)
        self.log('test_accuracy', accuracy)
        print('Test accuracy={}'.format(accuracy))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self['lr'])
