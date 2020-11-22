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


def collate_fn(batch):
    output = {
        'positive_ids': [],
        'negative_ids': [],
        'positive_samples': [],
        'negative_samples': []
    }
    for example in batch:
        output['positive_ids'] += example['positive_ids']
        output['negative_ids'] += example['negative_ids']
        output['positive_samples'].append(len(example['positive_ids']))
        output['negative_samples'].append(len(example['negative_ids']))
    output['positive_ids'] = torch.LongTensor(output['positive_ids'])
    output['negative_ids'] = torch.LongTensor(output['negative_ids'])
    return output


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
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self['lr'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for E-Grid model.')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--embedding_dim', default=100, type=int)  # paper is 300
    parser.add_argument('--num_maps', default=100, type=int)  # paper is 100
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('-mini', default=False, action='store_true')
    args = parser.parse_args()

    gpus = torch.cuda.device_count() if torch.cuda.is_available() and not args.cpu else None

    weights_dir = os.path.join('weights', args.experiment)
    print('Loading vocabulary...')
    with open('data/vocab.pk', 'rb') as fd:
        vocab = pickle.load(fd)

    if not os.path.exists(weights_dir):
        print('Creating {} path'.format(weights_dir))
        os.mkdir(weights_dir)

    # default used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=weights_dir,
        save_top_k=1,
        verbose=False,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    gpus = torch.cuda.device_count() if torch.cuda.is_available() and not args.cpu else None
    distributed_backend = None if gpus is None else 'ddp'

    train_dataset = EGridDataset(vocab, split='train', k=args.k, mini=args.mini)
    val_dataset = EGridDataset(vocab, split='validation', k=args.k, mini=args.mini)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    early_stopping = EarlyStopping('val_loss')

    trainer = pl.Trainer(
        callbacks=[early_stopping, checkpoint_callback],
        min_epochs=1,
        max_epochs=20,
        gpus=gpus,
        distributed_backend=distributed_backend,
        val_check_interval=0.2,
        deterministic=True,
        accumulate_grad_batches=1,
        terminate_on_nan=True,
    )

    model = EntityGridModel(args, vocab)
    trainer.fit(model, train_loader, val_loader)
