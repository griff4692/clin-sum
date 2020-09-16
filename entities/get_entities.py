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
from tqdm import tqdm
from p_tqdm import p_uimap

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.section_utils import resolve_course, sents_from_html
from preprocess.utils import *


blacklist_cui_name = [
    'present illness',
    'medical history',
    'admission',
    'discharge'
    'follow-up',
    'hospital course',
    'sitting position',
    'symptoms',
    'medical examination',
    'chief complaint',
]


whitelist_semgroup = {
    'Chemicals & Drugs',
    'Disorders',
    'Procedures',
}


def ok_cui_name(cui_name):
    cnl = cui_name.lower()
    for x in blacklist_cui_name:
        if x in cnl:
            return False
    return True


def _process(mrn, account, text, cui, is_t):
    is_core = cui in core_cui_set
    spacy_ent = cui_to_ent_map[cui]

    cui_name = spacy_ent[1]
    tuis = spacy_ent[3]
    groups = list(set([tui_group_map[tui] for tui in tuis]))
    definition = spacy_ent[4]

    matching_groups = [group for group in groups if group in whitelist_semgroup]
    is_relevant = (is_core or len(matching_groups) > 0) and ok_cui_name(cui_name)

    if not is_relevant:
        return None

    row = {
        'mrn': mrn,
        'account': account,
        'is_target': is_t,
        'is_source': not is_t,
        'extracted_text': text,
        'cui': cui,
        'is_core': is_core,
        'cui_name': cui_name,
        'tui': tuis[0],
        'sem_group': groups[0],
        'definition': definition,
        'other_tuis': None if len(tuis) == 1 else json.dumps(tuis[1:]),
        'other_groups': None if len(groups) == 1 else json.dumps(groups[1:]),
    }

    return row


def extract_entities(record):
    target_inventory = defaultdict(list)
    rows = []
    mrn = record['mrn']
    account = record['account']

    source_sents = sents_from_html(record['spacy_source_toks'], convert_lower=False)
    target_sents = sents_from_html(resolve_course(record['spacy_target_toks']), convert_lower=False)

    for source_sent in source_sents:
        for entity in spacy_nlp(source_sent).ents:
            kb_ents = entity._.kb_ents
            if len(kb_ents) == 0:
                continue
            cui = kb_ents[0][0]
            row = _process(mrn, account, entity.string.strip(), cui, False)
            if row is not None:
                rows.append(row)

    for target_sent in target_sents:
        for entity in spacy_nlp(target_sent).ents:
            kb_ents = entity._.kb_ents
            if len(kb_ents) == 0:
                continue
            cui = kb_ents[0][0]
            row = _process(mrn, account, entity.string.strip(), cui, True)
            if row is not None:
                rows.append(row)
                target_inventory[cui].append(target_sent)

    inventory_fn = os.path.join(out_dir, 'entity', 'inventory', '{}_{}.json'.format(str(mrn), str(account)))
    ents_fn = os.path.join(out_dir, 'entity', 'entities', '{}_{}.csv'.format(str(mrn), str(account)))
    with open(inventory_fn, 'w') as fd:
        json.dump(target_inventory, fd)

    df = pd.DataFrame(rows)
    df.to_csv(ents_fn, index=False)

    return len(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to extract and UMLS-link entities for all examples.')
    parser.add_argument('-mini', default=False, action='store_true')
    parser.add_argument('-collect', default=False, action='store_true')

    args = parser.parse_args()
    base_dir = os.path.join(out_dir, 'entity')
    entities_dir = os.path.join(base_dir, 'entities')
    inventory_dir = os.path.join(base_dir, 'inventory')

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

        inventory_fns = list(map(lambda x: os.path.join(inventory_dir, x), os.listdir(inventory_dir)))
        target_inventory = list(tqdm(map(lambda fn: json.load(open(fn)), inventory_fns), total=len(inventory_fns)))

        full_inventory = defaultdict(list)
        for dict in target_inventory:
            for cui, sents in dict.items():
                full_inventory[cui] += sents
        for k, v in full_inventory.items():
            full_inventory[k] = Counter(v)
        out_inventory_fn = os.path.join(base_dir, 'target_cui_sent_index.json')
        with open(out_inventory_fn, 'w') as fd:
            json.dump(full_inventory, fd)
    else:
        if not os.path.exists(base_dir):
            print('Creating {} dir'.format(base_dir))
            os.mkdir(base_dir)
            os.mkdir(entities_dir)
            os.mkdir(inventory_dir)

        mini_str = '_small' if args.mini else ''

        snomed_core_fn = '../data/SNOMEDCT_CORE_SUBSET_202005/SNOMEDCT_CORE_SUBSET_202005.txt'
        semgroups_fn = '../data/umls_semgroups.txt'

        cols = ['UMLS_CUI', 'SNOMED_FSN', 'SNOMED_CID']
        snomed_df = pd.read_csv(snomed_core_fn, delimiter='|')[cols]
        core_cui_set = set(snomed_df['UMLS_CUI'].tolist())

        # sem_group_acronym|sem_group|tui|tui_description
        sem_groups_df = pd.read_csv(semgroups_fn, delimiter='|').dropna()
        tui_group_map = dict(zip(sem_groups_df['tui'].tolist(), sem_groups_df['sem_group'].tolist()))

        print('Loading spacy...')
        spacy_nlp = spacy.load('en_core_sci_lg')
        abbreviation_pipe = AbbreviationDetector(spacy_nlp)
        spacy_nlp.add_pipe(abbreviation_pipe)

        print('Loading UMLS entity linker...')
        linker = EntityLinker(resolve_abbreviations=True, name='umls')
        cui_to_ent_map = linker.kb.cui_to_entity
        spacy_nlp.add_pipe(linker)
        print('Let\'s go get some entities...')

        splits = ['validation', 'train']
        examples = get_records(split=splits, mini=args.mini).to_dict('records')
        num_ents = np.array(list(p_uimap(extract_entities, examples)))

        print('An average of {} entities extracted per visit'.format(num_ents.mean()))
