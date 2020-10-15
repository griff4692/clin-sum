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
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from extractive.neusum.attention import Attention
from extractive.neusum.data_utils import collate_fn, test_collate_fn, mask_2D, SingleExtractionDataset
from extractive.neusum.vocab import Vocab
from preprocess.constants import out_dir
from utils import tens_to_np

MAX_GEN_SUM_SENTS = 25
MAX_GEN_SUM_TOK_CT = 165


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

        self.dropout = nn.Dropout(p=0.25)
        self.sum_aware = True # TODO - remember why I did this! self.args.max_curr_sum_sents > 0
        s_dim = args.hidden_dim * 8 if self.sum_aware else args.hidden_dim * 4
        self.scorer = nn.Sequential(
            self.dropout,
            nn.Linear(s_dim, s_dim),
            nn.Tanh(),
            self.dropout,
            nn.Linear(s_dim, s_dim),
            nn.Tanh(),
            self.dropout,
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

    def kld(self, y_hat_scores, y_dist):
        loss_func = nn.KLDivLoss(log_target=False, reduction='batchmean')
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

        score_input = source_h
        if self.sum_aware:
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

        y_mask = masks['source_mask']
        mask_max_trunc_idx = min(y_hat_scores.size()[1], y_mask.size()[1])
        y_mask = y_mask[:, :mask_max_trunc_idx]
        y_hat_scores.masked_fill_(y_mask, float('-inf'))

        if self.objective == 'bce':
            loss_val = self.bce(y_hat_scores, y_dist)
        else:
            loss_val = self.kld(y_hat_scores, y_dist)
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

    def test_step(self, batch, batch_idx):
        sent_order = []
        sum_ids = []
        sum_sent_lens = []
        sum_sent_toks = []
        sum_len = 0
        source_ids_flat_pad, sum_ids_flat_pad, target_dist, counts, masks, metadata = batch
        gold_sent_order = list(np.argsort(tens_to_np(-target_dist.squeeze())))
        num_sents = len(metadata['source_sents'][0])
        mrn = metadata['mrn'][0]
        rel_ranks = []
        account = metadata['account'][0]
        for _ in range(min(num_sents, MAX_GEN_SUM_SENTS)):
            i0 = source_ids_flat_pad.to(self.device_name)
            i1 = sum_ids_flat_pad.to(self.device_name)
            i2 = {}
            i3 = {}
            for k, v in counts.items():
                i2[k] = v.to(self.device_name)
            for k, v in masks.items():
                i3[k] = v.to(self.device_name)

            y_hat_scores = self(i0, i1, i2, i3)
            y_hat_scores = tens_to_np(y_hat_scores.squeeze(0))
            if len(sent_order) > 0:
                y_hat_scores[sent_order] = float('-inf')

            max_idx = np.argmax(y_hat_scores)
            rel_ranks.append(gold_sent_order.index(max_idx))
            sent_sum_len = counts['source_sent_lens_flat'][max_idx]
            sum_len += sent_sum_len
            if sum_len > MAX_GEN_SUM_TOK_CT:
                break

            sent_order.append(max_idx)
            chosen_sent_toks = metadata['source_sents'][0][max_idx]
            sum_sent_toks.append(chosen_sent_toks)
            num_sum_sents = len(sent_order)
            chosen_sent_ids = list(tens_to_np(source_ids_flat_pad[max_idx][:sent_sum_len]))
            sum_ids.append(chosen_sent_ids)
            sum_sent_lens.append(sent_sum_len)
            sum_ids_flat = list(map(torch.LongTensor, sum_ids))
            sum_ids_flat_pad = pad_sequence(sum_ids_flat, batch_first=True, padding_value=0)
            sum_att_mask = mask_2D([num_sum_sents])
            counts['sum_sent_lens_flat'] = torch.LongTensor(sum_sent_lens)
            counts['sum_lens'] = torch.LongTensor([len(sent_order)])
            masks['sum_att_mask'] = sum_att_mask
        result = pl.EvalResult()
        result.mrn = mrn
        result.account = account
        result.sent_order = ','.join([str(s) for s in sent_order])
        result.sum_sent_toks = ' <s> '.join(sum_sent_toks)
        result.reference = metadata['reference'][0]

        result.rel_r1 = rel_ranks[0]
        result.rel_r2 = rel_ranks[1]
        result.rel_r3 = rel_ranks[2]
        result.rel_r4 = rel_ranks[3]
        result.rel_r5 = rel_ranks[4]
        result.rel_r5plus = sum(rel_ranks[5:]) / float(len(rel_ranks[5:]))

        return result

    def avg(self, arr):
        return sum(arr) / float(len(arr))

    def test_epoch_end(self, test_outputs):
        mrns = test_outputs.mrn
        accounts = test_outputs.account
        sent_orders = test_outputs.sent_order
        sum_sents = test_outputs.sum_sent_toks
        references = test_outputs.reference

        output_df = {
            'mrn': mrns,
            'account': accounts,
            'sent_orders': sent_orders,
            'prediction': sum_sents,
            'reference': references
        }
        output_df = pd.DataFrame(output_df)
        exp_str = 'neusum_{}_{}'.format(self.args.experiment, str(MAX_GEN_SUM_TOK_CT))
        out_fn = os.path.join(out_dir, 'predictions', '{}_validation.csv'.format(exp_str))
        print('To evaluate, run: cd ../../evaluations && python rouge.py --experiment {}'.format(exp_str))

        output_df.to_csv(out_fn, index=False)

        eval_result = pl.EvalResult()
        eval_result.log('rank@1', self.avg(test_outputs.rel_r1), on_epoch=True)
        eval_result.log('rank@2', self.avg(test_outputs.rel_r2), on_epoch=True)
        eval_result.log('rank@3', self.avg(test_outputs.rel_r3), on_epoch=True)
        eval_result.log('rank@4', self.avg(test_outputs.rel_r4), on_epoch=True)
        eval_result.log('rank@5+', self.avg(test_outputs.rel_r5plus), on_epoch=True)
        return eval_result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self['lr'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for NeuSum extractive baseline.')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--embedding_dim', default=50, type=int)
    parser.add_argument('--hidden_dim', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('-mini', default=False, action='store_true')
    parser.add_argument('--objective', default='kld', choices=['bce', 'kld'])
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--q_temp', default=5.0, type=float, help='Temperature smoothing coefficient when taking '
                                                                  'softmax over true label distribution.'
                        )
    parser.add_argument('-eval_only', default=False, action='store_true')
    parser.add_argument('--max_eval', default=None, type=int)
    parser.add_argument('--max_curr_sum_sents', default=50, type=int)

    args = parser.parse_args()

    num_workers = int(0.75 * multiprocessing.cpu_count())
    gpus = torch.cuda.device_count() if torch.cuda.is_available() and not args.cpu else None
    if gpus is not None and args.batch_size % gpus > 0:
        raise Exception('Target batch size={} must be a multiple of number of GPUs={}'.format(args.batch_size, gpus))

    weights_dir = os.path.join('weights', args.experiment)

    print('Loading vocabulary...')
    with open('data/vocab_num_template.pk', 'rb') as fd:
        vocab = pickle.load(fd)
    vocab.add_token('<s>')

    if not args.eval_only:
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

        train_dataset = SingleExtractionDataset(
            vocab, type='train', mini=args.mini, label_temp=args.q_temp, max_curr_sum_sents=args.max_curr_sum_sents)
        val_dataset = SingleExtractionDataset(
            vocab, type='validation', mini=args.mini, label_temp=args.q_temp, max_curr_sum_sents=args.max_curr_sum_sents)
        test_dataset = SingleExtractionDataset(
            vocab, type='validation', mini=args.mini, max_curr_sum_sents=0, trunc=False, max_n=args.max_eval)

        per_device_batch_size = args.batch_size if gpus is None else args.batch_size // gpus
        if per_device_batch_size < args.batch_size:
            print('Setting per device batch size to {}.  Effective batch size is {}'.format(
                per_device_batch_size, args.batch_size))
        train_loader = DataLoader(
            train_dataset, batch_size=per_device_batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=per_device_batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=collate_fn
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=test_collate_fn)

        device_name = 'cpu' if gpus is None else 'cuda'
        precision = 16 if gpus is not None else 32
        distributed_backend = None if gpus is None else 'ddp'

        logger = None
        if not args.mini:
            logger = WandbLogger(name=args.experiment, save_dir=weights_dir, project='clinsum', log_model=False)

        early_stop_callback = None if args.mini else EarlyStopping(
            monitor='val_early_stop_on',
            min_delta=1e-4,
            patience=3,
            verbose=True,
            mode='min'
        )

        trainer = pl.Trainer(
            logger=logger,
            early_stop_callback=early_stop_callback,
            min_epochs=1,
            max_epochs=20,
            gpus=gpus,
            distributed_backend=distributed_backend,
            precision=precision,
            val_check_interval=0.2,
            deterministic=True,
            accumulate_grad_batches=1,
            auto_select_gpus=True,
            checkpoint_callback=checkpoint_callback,
            terminate_on_nan=True,
            # gradient_clip_val=0.5,
            # auto_lr_find=True,
            # auto_scale_batch_size='binsearch' will find largest batch size that fits into GPU memory
        )

        model = NeuSum(args, vocab, device_name=device_name)
        trainer.fit(model, train_loader, val_loader)
    else:
        checkpoint_fns = glob.glob(os.path.join(weights_dir, '*.ckpt'))
        assert len(checkpoint_fns) == 1
        model = NeuSum.load_from_checkpoint(checkpoint_fns[0], args=args, vocab=vocab)
        test_dataset = SingleExtractionDataset(
            model.vocab, type='validation', mini=args.mini, max_curr_sum_sents=0, trunc=False, max_n=args.max_eval,
            eval=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=test_collate_fn)
        trainer = pl.Trainer(
            gpus=1 if gpus is not None else 0,
            distributed_backend=None
        )
    trainer.test(model, test_dataloaders=test_loader)
