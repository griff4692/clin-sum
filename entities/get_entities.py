from datetime import datetime
from collections import defaultdict, Counter
from functools import partial
import os
import sys
from time import time

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.stats import describe
import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from p_tqdm import p_uimap

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.utils import *


def _process(mrn, account, name, cui, note_id, is_t):
    row = {
        'mrn': mrn,
        'account': account,
        'is_target': is_t,
        'is_source': not is_t,
        'note_id': note_id,
        'name': name,
        'cui': cui,
    }

    return row


def extract_entities(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    notes_fn = os.path.join(mrn_dir, 'notes.csv')
    entities_fn = os.path.join(mrn_dir, 'entities.csv')
    valid_accounts_fn = os.path.join(mrn_dir, 'valid_accounts.csv')
    if not os.path.exists(valid_accounts_fn):
        print('Not recognizing MRN={} as valid'.format(mrn))
        return

    valid_accounts = set(pd.read_csv(valid_accounts_fn)['account'].tolist())
    notes_df = pd.read_csv(notes_fn)
    valid_notes_df = notes_df[notes_df['account'].isin(valid_accounts)]
    source_notes = valid_notes_df[valid_notes_df['is_source']]
    target_notes = valid_notes_df[valid_notes_df['is_target']]
    num_source, num_target = len(source_notes), len(target_notes)
    assert min(num_source, num_target) > 0

    is_target = ([False] * num_source) + ([True] * num_target)
    records = source_notes.to_dict('records') + target_notes.to_dict('records')
    rows = []
    for i, record in enumerate(records):
        note_id = record['note_id']
        text = record['text']
        account = record['account']
        doc = spacy_nlp(text)
        for entity in doc.ents:
            kb_ents = entity._.kb_ents
            if len(kb_ents) == 0:
                continue
            cui = kb_ents[0][0]
            rows.append(_process(mrn, account, str(entity), cui, note_id, is_target[i]))

    df = pd.DataFrame(rows)
    # df = df[((~df['snomed_cid'].isnull()) | (df['umls_type'].isin(WHITELIST_UMLS_TYPES)))]
    entity_n = df.shape[0]
    if entity_n == 0:
        print('No entities found for MRN={}'.format(mrn))
    df.to_csv(entities_fn, index=False)
    return len(df)


if __name__ == '__main__':
    print('Loading spacy...')
    spacy_nlp = spacy.load('en_core_sci_lg')
    abbreviation_pipe = AbbreviationDetector(spacy_nlp)
    spacy_nlp.add_pipe(abbreviation_pipe)

    print('Loading UMLS entity linker...')
    linker = EntityLinker(resolve_abbreviations=True, name='umls')
    spacy_nlp.add_pipe(linker)
    print('Let\'s go get some entities...')
    _, _, mrns = get_mrn_status_df('valid_example')
    n = len(mrns)

    print('Processing {} mrns'.format(n))
    start_time = time()
    p_uimap(extract_entities, mrns, num_cpus=0.8)
    duration(start_time)
