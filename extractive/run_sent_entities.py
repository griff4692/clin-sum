from collections import Counter, defaultdict
from functools import partial
import itertools
import json
import math
from multiprocessing import Pool
import os
from string import punctuation
import sys

sys.path.insert(0, os.path.expanduser('~/clin-sum'))

from medcat.cat import CAT
from medcat.utils.vocab import Vocab
from medcat.cdb import CDB
import numpy as np
import pandas as pd
from tqdm import tqdm

import argparse
from evaluations.rouge import max_rouge_sent
from preprocess.constants import out_dir
from preprocess.section_utils import resolve_course, sents_from_html, sent_toks_from_html
from preprocess.utils import get_mrn_status_df

snomed_core_fn = '../data/SNOMEDCT_CORE_SUBSET_202005/SNOMEDCT_CORE_SUBSET_202005.txt'
core_cols = ['SNOMED_CID', 'SNOMED_FSN', 'SNOMED_CONCEPT_STATUS']
umls_cols = ['pretty_name', 'cui', 'tui', 'type', 'source_value', 'start', 'end', 'acc']


def sent_level_ents(mrn):
    example_fn = os.path.join(out_dir, 'mrn', mrn, 'examples.csv')
    example_df = pd.read_csv(example_fn)
    records = example_df.to_dict('records')
    outputs = []
    for record in records:
        target_sents = sents_from_html(resolve_course(record['spacy_target_toks']), convert_lower=False)
        target_sent_ents = []
        for target_sent in target_sents:
            entities = cat.get_entities(target_sent)
            rel_entities = []
            for entity in entities:
                cui = entity['cui']
                if cui in core_cuis:
                    rel_entities.append(cui)
            target_sent_ents.append(rel_entities)
        reference = ' <s> '.join(target_sents).strip()
        outputs.append({
            'account': record['account'],
            'mrn': record['mrn'],
            'reference': reference,
            'entities': json.dumps(target_sent_ents)
        })

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate oracle predictions')
    parser.add_argument('--max_n', default=-1, type=int)

    args = parser.parse_args()

    snomed_core_df = pd.read_csv(snomed_core_fn, sep='|')
    snomed_core_df = snomed_core_df[core_cols + ['UMLS_CUI']]
    snomed_core_df.dropna(subset=['UMLS_CUI'], inplace=True)
    core_cuis = set(snomed_core_df['UMLS_CUI'].tolist())

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

    splits_df = pd.read_csv(os.path.join(out_dir, 'splits.csv'))
    val_df = splits_df[splits_df['split'] == 'validation']
    val_mrns = val_df['mrn'].astype('str').unique().tolist()

    if args.max_n > 0:
        np.random.seed(1992)
        val_mrns = np.random.choice(val_mrns, size=args.max_n, replace=False)

    outputs = list(filter(None, tqdm(map(sent_level_ents, val_mrns), total=len(val_mrns))))
    outputs_flat = list(itertools.chain(*outputs))
    out_fn = os.path.join(out_dir, 'predictions', 'sent_entities_validation.csv')
    output_df = pd.DataFrame(outputs_flat)
    output_df.to_csv(out_fn, index=False)
