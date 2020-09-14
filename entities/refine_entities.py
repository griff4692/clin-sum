from collections import defaultdict
import json
import os
import sys

import pandas as pd
from p_tqdm import p_uimap
import spacy
import scispacy
from scispacy.linking import EntityLinker

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.utils import get_mrn_status_df


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


def refine_entities(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    entities_fn = os.path.join(mrn_dir, 'entities.csv')
    out_fn = os.path.join(mrn_dir, 'entities_relevant.csv')
    df = pd.read_csv(entities_fn)

    out_df = []
    for row in df.to_dict('records'):
        cui = row['cui']
        is_core = cui in core_cui_set
        spacy_ent = cui_to_ent_map[cui]

        cui_name = spacy_ent[1]
        tuis = spacy_ent[3]
        groups = list(set([tui_group_map[tui] for tui in tuis]))
        definition = spacy_ent[4]

        matching_groups = [group for group in groups if group in whitelist_semgroup]
        is_relevant = (is_core or len(matching_groups) > 0) and ok_cui_name(cui_name)

        if is_relevant:
            meta_obj = {
                'is_core': is_core,
                'cui_name': spacy_ent[1],
                'tui': tuis[0],
                'sem_group': groups[0],
                'definition': definition,
                'other_tuis': None if len(tuis) == 1 else json.dumps(tuis[1:]),
                'other_groups': None if len(groups) == 1 else json.dumps(groups[1:]),
            }

            meta_obj.update(row)
            out_df.append(meta_obj)

    out_df = pd.DataFrame(out_df)
    out_df.to_csv(out_fn, index=False)

    return len(df), len(out_df)


if __name__ == '__main__':
    snomed_core_fn = '../data/SNOMEDCT_CORE_SUBSET_202005/SNOMEDCT_CORE_SUBSET_202005.txt'
    semgroups_fn = '../data/umls_semgroups.txt'

    cols = ['UMLS_CUI', 'SNOMED_FSN', 'SNOMED_CID']
    snomed_df = pd.read_csv(snomed_core_fn, delimiter='|')[cols]
    core_cui_set = set(snomed_df['UMLS_CUI'].tolist())

    # sem_group_acronym|sem_group|tui|tui_description
    sem_groups_df = pd.read_csv(semgroups_fn, delimiter='|').dropna()
    tui_group_map = dict(zip(sem_groups_df['tui'].tolist(), sem_groups_df['sem_group'].tolist()))

    linker = EntityLinker(resolve_abbreviations=True, name='umls')

    cui_to_ent_map = linker.kb.cui_to_entity

    print('Let\'s go get some entities...')
    _, _, mrns = get_mrn_status_df('valid_example')
    n = len(mrns)

    print('Processing {} mrns'.format(n))
    outputs = list(p_uimap(refine_entities, mrns, num_cpus=0.77))

    orig_n = sum(x[0] for x in outputs)
    refined_n = sum(x[1] for x in outputs)

    print('Entity count decreased from {} to {}!'.format(orig_n, refined_n))
