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

import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

from medcat.cat import CAT
from medcat.utils.vocab import Vocab
from medcat.cdb import CDB

from egrids.data_utils import EGridDataset
from egrids.model import EntityGridModel
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for E-Grid model.')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('-build_egrid', default=False, action='store_true')
    parser.add_argument('-eval', default=False, action='store_true')
    parser.add_argument('--eval_fp', default='/nlp/projects/clinsum/predictions/oracle_greedy_rel_validation.csv')
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--embedding_dim', default=100, type=int)  # paper is 300
    parser.add_argument('--num_maps', default=100, type=int)  # paper is 100
    parser.add_argument('--k', default=5, type=int)
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

    gpus = torch.cuda.device_count() if torch.cuda.is_available() and not args.cpu else None
    distributed_backend = None if gpus is None else 'ddp'

    # default used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=weights_dir,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],  # , early_stopping],
        min_epochs=1,
        max_epochs=10,
        gpus=gpus,
        distributed_backend=distributed_backend,
        val_check_interval=0.5,
        deterministic=True,
        accumulate_grad_batches=1,
        terminate_on_nan=True,
    )

    if args.build_egrid:
        vocab = Vocab()
        print('Loading vocabulary...')
        # Load the vocab model you downloaded
        vocab.load_dict('../data/medcat/vocab.dat')

        # Load the cdb model you downloaded
        cdb = CDB()
        print('Loading model...')
        cdb.load_dict('../data/medcat/cdb.dat')

        # create cat
        print('Creating MedCAT pipeline...')
        cat = CAT(cdb=cdb, vocab=vocab)

        print('Loading UMLS entity linker...')
        linker = EntityLinker(resolve_abbreviations=True, name='umls')
        cui_to_ent_map = linker.kb.cui_to_entity

        eval_df = pd.read_csv(args.eval_fp)
        records = eval_df.to_dict('records')
        egrids = []
        for record in records:
            prediction = record['predictions']
            egrid = defaultdict(list)
            sents = prediction.split(' <s> ')
            for sent_idx, sent in enumerate(sents):
                ents = cat.get_entities(sents[sent_idx])
                for entity in entities:
                    is_core = entity['cui'] in core_cui_set
                    if entity['tui'] is None or entity['tui'] == 'None':
                        sem_group = None
                    else:
                        sem_group = tui_group_map[entity['tui']]
                    is_relevant = sem_group is None or is_core or sem_group in whitelist_semgroup or entity[
                        'tui'] in whitelist_tuis
                    if is_relevant:
                        # output
                        egrid[entity['cui']].append({
                            'source_value': entity['source_value'],
                            'sent_idx': sent_idx,
                        })
            egrids.append({'egrid': egrid, 'mrn': record['mrn'], 'split': 'validation', 'num_target_sents': len(sents)})
            with open(os.path.join(out_dir, 'egrids_test.json', 'w')) as fd:
                json.dump(egrids, fd)
    elif args.eval:
        weights_dir = 'weights/11_21_submit/'
        fn = weights_dir + os.listdir(weights_dir)[0]
        model = EntityGridModel.load_from_checkpoint(checkpoint_path=fn, args=args, vocab=vocab)

        eval_dataset = EGridDataset(vocab, split='eval', k=args.k, egrids=egrids)
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
        )
        trainer.test(model, eval_loader)
    else:
        train_dataset = EGridDataset(vocab, split='train', k=args.k, mini=args.mini)
        val_dataset = EGridDataset(vocab, split='validation', k=args.k, mini=args.mini)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        early_stopping = EarlyStopping('val_loss')

        model = EntityGridModel(args, vocab)
        trainer.fit(model, train_loader, val_loader)
