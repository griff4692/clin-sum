from collections import OrderedDict
import os
import multiprocessing
import pickle
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from extractive.neusum.attention import Attention
from extractive.neusum.data_utils import collate_fn, SingleExtractionDataset
from extractive.neusum.vocab import Vocab


class NeuSum(pl.LightningModule):
    def __init__(self, args, vocab, device_name='cuda'):
        super(NeuSum, self).__init__()
        self.vocab = vocab
        self.args = args
        self.embedding = nn.Embedding(len(vocab), embedding_dim=args.embedding_dim)
        self.source_sent_encoder = nn.LSTM(
            args.embedding_dim, hidden_size=args.hidden_dim, bidirectional=True, batch_first=True
        )
        self.source_doc_encoder = nn.LSTM(
            args.hidden_dim * 2, hidden_size=args.hidden_dim, bidirectional=True, batch_first=True
        )

        self.sum_sent_encoder = nn.LSTM(
            args.embedding_dim, hidden_size=args.hidden_dim, bidirectional=True, batch_first=True
        )

        self.sum_doc_encoder = nn.LSTM(
            args.hidden_dim * 2, hidden_size=args.hidden_dim, bidirectional=True, batch_first=True
        )

        s_dim = args.hidden_dim * 8
        self.scorer = nn.Sequential(
            nn.Linear(s_dim, s_dim),
            nn.Tanh(),
            nn.Linear(s_dim, s_dim),
            nn.Tanh(),
            nn.Linear(s_dim, 1),
        )

        att_dim = args.hidden_dim * 4
        self.sum_aware_att = Attention(att_dim)

        self.objective = args.objective
        self.device_name = device_name

    def on_epoch_start(self):
        print('\n')

    def bce(self, y_hat_scores, y_dist):
        loss_func = nn.CrossEntropyLoss()
        y_amax = y_dist.argmax(dim=1)
        return loss_func(y_hat_scores, y_amax)

    def kld(self, y_hat_scores, y_dist, y_mask=None):
        mask_max_trunc_idx = min(y_hat_scores.size()[1], y_mask.size()[1])
        y_mask = y_mask[:, :mask_max_trunc_idx]
        loss_func = nn.KLDivLoss(log_target=False, reduction='batchmean')
        y_hat_scores.masked_fill_(y_mask, float('-inf'))
        y_hat_lprob = F.log_softmax(y_hat_scores, dim=-1)
        return loss_func(y_hat_lprob, y_dist)

    def __getitem__(self, item):
        return getattr(self.args, item)

    def hier_encode(self, ids, sent_lens, seq_lens, sent_rnn, doc_rnn):
        n = len(ids)
        embeds = self.embedding(ids)
        packed_embeds = pack_padded_sequence(embeds, sent_lens, batch_first=True, enforce_sorted=False)
        _, sent_h_flat = sent_rnn(packed_embeds)
        if type(sent_h_flat) == tuple:
            sent_h_flat = sent_h_flat[0]
        sent_h_flat = sent_h_flat.transpose(1, 0).contiguous().view(n, -1)

        sent_h = []
        start_idx = 0
        for seq_len in seq_lens:
            sent_h.append(sent_h_flat[start_idx:start_idx + seq_len])
            start_idx += seq_len
        padded_sent_h = pad_sequence(sent_h, batch_first=True, padding_value=0)

        docs_packed = pack_padded_sequence(
            padded_sent_h, seq_lens, batch_first=True, enforce_sorted=False)
        sent_modeled_packed_h, _ = doc_rnn(docs_packed)
        sent_modeled_h, _ = pad_packed_sequence(sent_modeled_packed_h, batch_first=True)

        joint_output = torch.cat([padded_sent_h, sent_modeled_h], dim=-1)
        return joint_output

    def forward(self, source_ids, sum_ids, counts, masks):
        source_h = self.hier_encode(
            source_ids, counts['source_sent_lens_flat'], counts['source_lens'], self.source_sent_encoder,
            self.source_doc_encoder
        )

        sum_h = self.hier_encode(
            sum_ids, counts['sum_sent_lens_flat'], counts['sum_lens'], self.sum_sent_encoder,
            self.sum_doc_encoder
        )

        sum_aware_source_h, _ = self.sum_aware_att(source_h, sum_h, masks['sum_att_mask'])
        score_input = torch.cat([source_h, sum_aware_source_h], dim=2)
        scores = self.scorer(score_input).squeeze(-1)
        return scores

    def shared_step(self, batch):
        source_ids_flat_pad, sum_ids_flat_pad, y_dist, counts, masks = batch
        y_hat_scores = self(source_ids_flat_pad, sum_ids_flat_pad, counts, masks)

        if self.objective == 'bce':
            loss_val = self.bce(y_hat_scores, y_dist)
        else:
            loss_val = self.kld(y_hat_scores, y_dist, y_mask=masks['source_mask'])
        return loss_val

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self['lr'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for NeuSum extractive baseline.')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--embedding_dim', default=50, type=int)
    parser.add_argument('--hidden_dim', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('-mini', default=False, action='store_true')
    parser.add_argument('--objective', default='bce', choices=['bce', 'kld'])
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--q_temp', default=1.0, type=float, help='Temperature smoothing coefficient when taking '
                                                                  'softmax over true label distribution.'
                        )

    args = parser.parse_args()

    weights_dir = os.path.join('weights', args.experiment)
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

    print('Loading vocabulary...')
    with open('data/vocab_num_template.pk', 'rb') as fd:
        vocab = pickle.load(fd)
    vocab.add_token('<s>')
    train_dataset = SingleExtractionDataset(vocab, type='train', mini=args.mini, label_temp=args.q_temp)
    val_dataset = SingleExtractionDataset(vocab, type='validation', mini=args.mini)
    num_workers = int(0.5 * multiprocessing.cpu_count())
    gpus = torch.cuda.device_count() if torch.cuda.is_available() and not args.cpu else None

    if gpus is not None and args.batch_size % gpus > 0:
        raise Exception('Target batch size={} must be a multiple of number of GPUs={}'.format(args.batch_size, gpus))

    per_device_batch_size = args.batch_size if gpus is None else args.batch_size // gpus
    if per_device_batch_size < args.batch_size:
        print('Setting per device batch size to {}.  Effective batch size is {}'.format(
            per_device_batch_size, args.batch_size))
    train_loader = DataLoader(
        train_dataset, batch_size=per_device_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=per_device_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    device_name = 'cpu' if gpus is None else 'cuda'
    precision = 16 if gpus is not None else 32
    distributed_backend = None if gpus is None else 'ddp'
    wandb_logger = WandbLogger(name=args.experiment, save_dir=weights_dir, project='clinsum', log_model=False)

    early_stop_callback = EarlyStopping(
        monitor='val_early_stop_on',
        min_delta=1e-3,
        patience=2,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        early_stop_callback=early_stop_callback,
        min_epochs=1,
        max_epochs=20,
        gpus=gpus,
        distributed_backend=distributed_backend,
        precision=precision,
        check_val_every_n_epoch=1,
        deterministic=True,
        accumulate_grad_batches=1,
        auto_select_gpus=True,
        checkpoint_callback=checkpoint_callback,
        terminate_on_nan=True,
        # auto_lr_find=True,
        # auto_scale_batch_size='binsearch' will find largest batch size that fits into GPU memory
    )

    model = NeuSum(args, vocab, device_name=device_name)
    trainer.fit(model, train_loader, val_loader)
