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
    return {
        'positive_ids': torch.LongTensor(batch[0]['positive_ids']),
        'negative_ids': torch.LongTensor(batch[0]['negative_ids']),
    }


class EntityGridModel(pl.LightningModule):
    def __init__(self, args, vocab, device_name='cuda'):
        super(EntityGridModel, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(len(vocab), embedding_dim=args.embedding_dim)

        self.maps = [
            nn.Sequential(
                nn.Linear(self['k'] * args.embedding_dim, 1),
                nn.ReLU()
            ) for _ in range(self['num_maps'])
        ]

        self.output = nn.Linear(self['num_maps'], 1)
        self.device_name = device_name

    def on_epoch_start(self):
        print('\n')

    def __getitem__(self, item):
        return getattr(self.args, item)

    def forward(self, ids):
        return scores

    def shared_step(self, batch):
        num_windows, window_size = batch.size()
        embeds = self.embedding(batch)
        embeds_concat = embeds.view(num_windows, -1)
        fmaps = torch.cat([
            mapper(embeds_concat) for mapper in self.maps
        ], dim=1)
        max_feats, _ = torch.max(fmaps, dim=0)
        return self.output(max_feats)

    def training_step(self, batch, batch_idx):
        pos_score = self.shared_step(batch['positive_ids'])
        neg_score = self.shared_step(batch['negative_ids'])
        loss = 1 - pos_score + neg_score
        loss.clamp_min_(0)
        return torch.clamp(loss, min=0)

    def validation_step(self, batch, batch_idx):
        pos_score = self.shared_step(batch['positive_ids'])
        neg_score = self.shared_step(batch['negative_ids'])
        loss = 1 - pos_score + neg_score
        return torch.clamp(loss, min=0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self['lr'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for E-Grid model.')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--embedding_dim', default=300, type=int)
    parser.add_argument('--num_maps', default=150, type=int)
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
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

    train_dataset = EGridDataset(vocab, split='train', k=args.k)
    val_dataset = EGridDataset(vocab, split='validation', k=args.k)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device_name = 'cpu' if gpus is None else 'cuda'
    precision = 16 if gpus is not None else 32
    distributed_backend = None if gpus is None else 'ddp'

    early_stopping = EarlyStopping('val_loss')

    trainer = pl.Trainer(
        callbacks=[early_stopping, checkpoint_callback],
        min_epochs=1,
        max_epochs=20,
        gpus=gpus,
        distributed_backend=distributed_backend,
        precision=precision,
        val_check_interval=0.2,
        deterministic=True,
        accumulate_grad_batches=args.batch_size,
        auto_select_gpus=True,
        terminate_on_nan=True,
    )

    model = EntityGridModel(args, vocab, device_name=device_name)
    trainer.fit(model, train_loader, val_loader)
