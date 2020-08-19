from collections import OrderedDict
import os
import multiprocessing
import pickle
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import argparse
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from extractive.neusum.data_utils import collate_fn, SingleExtractionDataset
from extractive.neusum.vocab import Vocab
from preprocess.generate_extractive_mmr_samples import Example


class NeuSum(pl.LightningModule):
    def __init__(self, args, vocab):
        super(NeuSum, self).__init__()
        self.vocab = vocab
        self.args = args
        self.embedding = nn.Embedding(len(vocab), embedding_dim=args.embedding_dim)
        self.source_sent_encoder = nn.GRU(
            args.embedding_dim, hidden_size=args.hidden_dim, bidirectional=False, batch_first=True
        )
        self.source_doc_encoder = nn.GRU(
            args.hidden_dim, hidden_size=args.hidden_dim, bidirectional=False, batch_first=True
        )

        self.sum_sent_encoder = nn.GRU(
            args.embedding_dim, hidden_size=args.hidden_dim, bidirectional=False, batch_first=True
        )

        self.sum_doc_encoder = nn.GRU(
            args.hidden_dim, hidden_size=args.hidden_dim, bidirectional=False, batch_first=True
        )

        s_dim = args.hidden_dim * 2
        self.scorer = nn.Sequential(
            nn.Linear(s_dim, s_dim),
            nn.Tanh(),
            nn.Linear(s_dim, s_dim),
            nn.Tanh(),
            nn.Linear(s_dim, 1),
        )

        self.softmax = nn.Softmax(dim=1)

    def __getitem__(self, item):
        return getattr(self.args, item)

    def hier_encode(self, ids, sent_lens, seq_lens, sent_rnn, doc_rnn):
        embeds = self.embedding(ids)
        packed_embeds = pack_padded_sequence(embeds, sent_lens, batch_first=True, enforce_sorted=False)
        _, sent_h_flat = sent_rnn(packed_embeds)
        sent_h_flat = sent_h_flat.squeeze(0)

        sent_h = []
        start_idx = 0
        for seq_len in seq_lens:
            sent_h.append(sent_h_flat[start_idx:start_idx + seq_len])
            start_idx += seq_len
        padded_sent_h = pad_sequence(sent_h, batch_first=True, padding_value=0)

        docs_packed = pack_padded_sequence(
            padded_sent_h, seq_lens, batch_first=True, enforce_sorted=False)
        sent_modeled_packed_h, _ = doc_rnn(docs_packed)
        sent_modeled_h, doc_h = pad_packed_sequence(sent_modeled_packed_h, batch_first=True)

        joint_output = torch.cat([padded_sent_h, sent_modeled_h], dim=-1)
        return joint_output, doc_h

    def forward(self, source_ids, sum_ids, counts):
        source_h, _ = self.hier_encode(
            source_ids, counts['source_sent_lens_flat'], counts['source_lens'], self.source_sent_encoder,
            self.source_doc_encoder
        )

        # _, doc_h = self.hier_encode(
        #     sum_ids, counts['sum_sent_lens_flat'], counts['sum_lens'], self.sum_sent_encoder,
        #     self.sum_doc_encoder
        # )

        scores = self.scorer(source_h).squeeze(-1)
        score_dist = self.softmax(scores)
        return score_dist

    def training_step(self, batch, batch_idx):
        source_ids_flat_pad, sum_ids_flat_pad, y, counts = batch
        y_hat = self(source_ids_flat_pad, sum_ids_flat_pad, counts)
        loss_val = F.kl_div(y_hat, y, log_target=False, reduction='batchmean')
        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict(
            {'loss': loss_val, 'progress_bar': tqdm_dict, 'log': tqdm_dict}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self['lr'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for NeuSum extractive baseline.')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--embedding_dim', default=50, type=int)
    parser.add_argument('--hidden_dim', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('-cpu', default=False, action='store_true')

    args = parser.parse_args()
    print('Loading vocabulary...')
    with open('data/vocab_num_template.pk', 'rb') as fd:
        vocab = pickle.load(fd)
    print('Constructing dataset')
    model = NeuSum(args, vocab)
    train_dataset = SingleExtractionDataset(vocab)
    num_workers = int(0.5 * multiprocessing.cpu_count())
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    gpus = torch.cuda.device_count() if torch.cuda.is_available() and not args.cpu else None
    precision = 16 if gpus is not None else 32
    trainer = pl.Trainer(gpus=gpus, precision=precision)
    trainer.fit(model, train_loader)
