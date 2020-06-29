from datetime import datetime
from collections import defaultdict, Counter
from functools import partial
import os
from multiprocessing import Manager, Pool, Value
from time import time, sleep
import warnings

from medcat.cat import CAT
from medcat.utils.vocab import Vocab
from medcat.cdb import CDB
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.stats import describe
from tqdm import tqdm

notes_dir = '/nlp/projects/clinsum/notes_by_mrn'
snomed_core_fn = '../data/SNOMEDCT_CORE_SUBSET_202005/SNOMEDCT_CORE_SUBSET_202005.txt'
core_cols = ['SNOMED_CID', 'SNOMED_FSN', 'SNOMED_CONCEPT_STATUS']
umls_cols = ['pretty_name', 'cui', 'tui', 'type', 'source_value', 'start', 'end', 'acc']


def _process(mrn, entity, note_id, is_t):
    row = {
        'mrn': mrn,
        'is_target': is_t,
        'is_source': not is_t,
        'note_id': note_id,
        'pretty_name': entity['pretty_name'],
        'cui': entity['cui'],
        'tui': entity['tui'],
        'umls_type': entity['type'],
        'source_value': entity['source_value'],
        'medcat_acc': entity['acc']
    }

    snomed_info = snomed_core_df[snomed_core_df['UMLS_CUI'] == row['cui']]
    snomed_records = snomed_info.to_dict('records')
    snomed_record = snomed_records[0] if len(snomed_records) > 0 else {}
    for c in core_cols:
        row[c.lower()] = snomed_record[c] if c in snomed_record else None
    return row


def extract_entities(mrn, mrn_counter=None, entity_counter=None, rel_entity_counter=None, lock=None):
    mrn_dir = os.path.join(notes_dir, mrn)
    notes_fn = os.path.join(mrn_dir, 'notes.csv')
    entities_fn = os.path.join(mrn_dir, 'entities.csv')

    try:
        notes_df = pd.read_csv(notes_fn)
    except pd.errors.EmptyDataError:
        print('Empty notes DataFrame!')
        return

    src_notes = notes_df[notes_df['is_rel_source']]
    dsum_notes = notes_df[notes_df['is_dsum']]
    is_target = ([False] * len(src_notes)) + ([True] * len(dsum_notes))
    records = src_notes.to_dict('records') + dsum_notes.to_dict('records')
    rows = []
    for i, record in enumerate(records):
        note_id = record['note_id']
        text = record['text']
        entities = cat.get_entities(text)
        rows += list(map(lambda entity: _process(mrn, entity, note_id, is_target[i]), entities))

    entity_n = len(rows)
    if entity_n > 0:
        df = pd.DataFrame(rows)
        entity_rel_n = df[~df['snomed_cid'].isnull()].shape[0]
        df.to_csv(entities_fn, index=False)
        with lock:
            mrn_counter.value += 1
            entity_counter.value += entity_n
            rel_entity_counter.value += entity_rel_n
            if mrn_counter.value % 100 == 0:
                print('Saved {} relevant entities for {} MRNs. {} total'.format(
                    rel_entity_counter.value, mrn_counter.value, entity_counter.value))
    else:
        assert not os.path.exists(os.path.join(mrn_dir, 'examples.csv'))


if __name__ == '__main__':
    snomed_core_df = pd.read_csv(snomed_core_fn, sep='|')
    snomed_core_df = snomed_core_df[core_cols + ['UMLS_CUI']]
    snomed_core_df.dropna(subset=['UMLS_CUI'], inplace=True)

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

    mrns = os.listdir(notes_dir)
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    with Manager() as manager:
        pool = Pool()  # By default pool will size depending on cores available
        mrn_counter = manager.Value('i', 0)
        entity_counter = manager.Value('i', 0)
        rel_entity_counter = manager.Value('i', 0)
        lock = manager.Lock()
        pool.map(
            partial(
                extract_entities, mrn_counter=mrn_counter, entity_counter=entity_counter,
                rel_entity_counter=rel_entity_counter, lock=lock
            ), mrns)
        pool.close()
        pool.join()

    end_time = time()
    minutes = (end_time - start_time) / 60.0
    round_factor = 0
    if minutes < 1:
        round_factor = 2
    print('Took {} minutes'.format(minutes, round(round_factor)))
