from datetime import datetime
from collections import defaultdict, Counter
import json
from functools import partial
import itertools
import os
import sys
from time import time

import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.stats import describe
import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

from medcat.cat import CAT
from medcat.utils.vocab import Vocab
from medcat.cdb import CDB

from tqdm import tqdm
from p_tqdm import p_uimap

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.section_utils import resolve_course, paragraph_from_html
from preprocess.tokenize_mrns import sent_segment
from preprocess.utils import *


whitelist_tuis = {
    'T034'  # Lab results
}

whitelist_semgroup = {
    'Chemicals & Drugs',
    'Disorders',
    'Procedures',
}


def generate_counts(account):
    outputs = []

    account_df = ent_df[ent_df['account'] == account]
    mrn = account_df['mrn'].iloc[0]
    source_df = account_df[account_df['is_source']]
    target_df = account_df[account_df['is_target']]

    source_cui_counts = source_df['cui'].value_counts().to_dict()
    target_cui_counts = target_df['cui'].value_counts().to_dict()

    all_cuis = list(set(list(source_cui_counts.keys()) + list(target_cui_counts.keys())))
    for cui in all_cuis:
        source_count = source_cui_counts.get(cui, 0)
        target_count = target_cui_counts.get(cui, 0)
        outputs.append({
            'mrn': mrn,
            'account': account,
            'cui': cui,
            'source_count': source_count,
            'target_count': target_count,
        })

    return outputs


def _process(mrn, account, entity, is_t, sent_idx):
    is_core = entity['cui'] in core_cui_set
    if entity['tui'] is None or entity['tui'] == 'None':
        sem_group = None
    else:
        sem_group = tui_group_map[entity['tui']]
    is_relevant = sem_group is None or is_core or sem_group in whitelist_semgroup or entity['tui'] in whitelist_tuis
    row = {
        'mrn': mrn,
        'account': account,
        'is_target': is_t,
        'is_source': not is_t,
        'cui': entity['cui'],
        'cui_name': entity['pretty_name'],
        'is_core': is_core,
        'tui': entity['tui'],
        'sem_group': sem_group,
        'umls_type': entity['type'],
        'source_value': entity['source_value'],
        'medcat_acc': entity['acc'],
        'sent_idx': sent_idx
    }

    if not is_relevant:
        return None

    return row


def extract_entities(record):
    rows = []
    mrn = record['mrn']
    account = record['account']

    source_paragraphs = [sent_segment(t, sentencizer=sentencizer) for t in paragraph_from_html(record['source_str'])]
    target_paragraphs = [sent_segment(t, sentencizer=sentencizer) for t in paragraph_from_html(record['target_str'])]
    assert len(source_paragraphs) >= 1 and len(target_paragraphs) >= 1
    source_sents = list(itertools.chain(*source_paragraphs))
    target_sents = list(itertools.chain(*target_paragraphs))

    num_source_sents = len(source_sents)
    num_target_sents = len(target_sents)

    sent_num = list(range(num_source_sents)) + list(range(num_target_sents))
    is_target = ([False] * num_source_sents) + ([True] * num_target_sents)
    all_sents = source_sents + target_sents

    for i, sent in enumerate(all_sents):
        entities = cat.get_entities(sent)
        sent_idx = sent_num[i]
        for entity in entities:
            row = _process(mrn, account, entity, is_target[i], sent_idx)
            if row is not None:
                rows.append(row)

    ents_fn = os.path.join(entities_dir, '{}_{}.csv'.format(str(mrn), str(account)))
    df = pd.DataFrame(rows)
    df.to_csv(ents_fn, index=False)

    return len(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to extract and UMLS-link entities for all examples.')
    parser.add_argument('-mini', default=False, action='store_true')
    parser.add_argument('-aggregate', default=False, action='store_true')
    parser.add_argument('-collect', default=False, action='store_true')
    parser.add_argument('-run', default=False, action='store_true')

    args = parser.parse_args()
    base_dir = os.path.join(out_dir, 'entity')
    entities_dir = os.path.join(base_dir, 'entities')

    if args.run:
        if not os.path.exists(base_dir):
            print('Creating {} dir'.format(base_dir))
            os.mkdir(base_dir)
        if not os.path.exists(entities_dir):
            print('Creating {} dir'.format(entities_dir))
            os.mkdir(entities_dir)

        mini_str = '_small' if args.mini else ''
        snomed_core_fn = '../data/SNOMEDCT_CORE_SUBSET_202005/SNOMEDCT_CORE_SUBSET_202005.txt'
        semgroups_fn = '../data/umls_semgroups.txt'

        cols = ['UMLS_CUI', 'SNOMED_FSN', 'SNOMED_CID']
        snomed_df = pd.read_csv(snomed_core_fn, delimiter='|')[cols]
        core_cui_set = set(snomed_df['UMLS_CUI'].tolist())

        # sem_group_acronym|sem_group|tui|tui_description
        sem_groups_df = pd.read_csv(semgroups_fn, delimiter='|').dropna()
        tui_group_map = dict(zip(sem_groups_df['tui'].tolist(), sem_groups_df['sem_group'].tolist()))

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

        print('Loading Spacy...')
        sentencizer = spacy.load('en_core_sci_lg', disable=['tagger', 'parser', 'ner', 'textcat'])
        sentencizer.add_pipe(sentencizer.create_pipe('sentencizer'))
        print('Loading UMLS entity linker...')
        linker = EntityLinker(resolve_abbreviations=True, name='umls')
        cui_to_ent_map = linker.kb.cui_to_entity
        print('Let\'s go get some entities...')

        splits = ['validation', 'train']
        examples = get_records(split=splits, mini=args.mini).to_dict('records')

        num_ents = np.array(list(p_uimap(extract_entities, examples, num_cpus=0.8)))
        print('An average of {} entities extracted per visit'.format(num_ents.mean()))

    if args.collect:
        ent_fns = list(map(lambda x: os.path.join(entities_dir, x), os.listdir(entities_dir)))
        print('Collecting {} different entity files.'.format(len(ent_fns)))
        ent_df_arr = []
        for i in tqdm(range(len(ent_fns))):
            try:
                ent_df_arr.append(pd.read_csv(ent_fns[i]))
            except pd.errors.EmptyDataError:
                print('No entities for {}'.format(ent_fns[i]))
        ent_df = pd.concat(ent_df_arr)
        out_ent_fn = os.path.join(base_dir, 'full_entities.csv')
        print('Saving {} entities to {}'.format(len(ent_df), out_ent_fn))
        ent_df.to_csv(out_ent_fn, index=False)
    if args.aggregate:
        in_ent_fn = os.path.join(base_dir, 'full_entities.csv')
        ent_df = pd.read_csv(in_ent_fn)
        accounts = ent_df['account'].unique().tolist()
        num_mrns = len(ent_df['mrn'].unique())
        print('Collected {} entities for {} unique visits across {} patients'.format(
            len(ent_df), len(accounts), num_mrns))
        outputs = list(p_uimap(generate_counts, accounts, num_cpus=0.8))
        outputs_flat = list(itertools.chain(*outputs))

        output_df = pd.DataFrame(outputs_flat)
        aggregated_out_fn = os.path.join(out_dir, 'entity', 'full_entities_aggregated.csv')
        n = len(output_df)
        output_df.sort_values(by=['source_count', 'target_count'], ascending=False, inplace=True)

        print('Saving {} CUI examples to {}'.format(n, aggregated_out_fn))
        output_df.to_csv(aggregated_out_fn, index=False)
