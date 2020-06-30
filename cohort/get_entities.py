from datetime import datetime
from collections import defaultdict, Counter
from functools import partial
import os
from multiprocessing import Manager, Pool, Value
from time import time

from medcat.cat import CAT
from medcat.utils.vocab import Vocab
from medcat.cdb import CDB
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.stats import describe
from tqdm import tqdm

from constants import *
from utils import *

snomed_core_fn = '../data/SNOMEDCT_CORE_SUBSET_202005/SNOMEDCT_CORE_SUBSET_202005.txt'
core_cols = ['SNOMED_CID', 'SNOMED_FSN', 'SNOMED_CONCEPT_STATUS']
umls_cols = ['pretty_name', 'cui', 'tui', 'type', 'source_value', 'start', 'end', 'acc']


def _process(mrn, account, title, entity, note_id, is_t):
    row = {
        'mrn': mrn,
        'account': account,
        'title': title,
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


def extract_entities(mrn, mrn_counter=None, entity_counter=None, lock=None):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    notes_fn = os.path.join(mrn_dir, 'notes.csv')
    entities_fn = os.path.join(mrn_dir, 'entities.csv')

    if os.path.exists(entities_fn):
        with lock:
            mrn_counter.value += 1
            if mrn_counter.value % 100 == 0:
                print('Saved {} relevant entities for {} MRNs'.format(entity_counter.value, mrn_counter.value))
            return

    notes_df = pd.read_csv(notes_fn)
    source_notes = notes_df[notes_df['is_source']]
    target_notes = notes_df[notes_df['is_target']]
    num_source, num_target = len(source_notes), len(target_notes)
    assert min(num_source, num_target) > 0

    is_target = ([False] * num_source) + ([True] * num_target)
    records = source_notes.to_dict('records') + target_notes.to_dict('records')
    rows = []
    for i, record in enumerate(records):
        note_id = record['note_id']
        text = record['text']
        account = record['account']
        title = record['title']
        entities = cat.get_entities(text)
        rows += list(map(lambda entity: _process(mrn, account, title, entity, note_id, is_target[i]), entities))

    df = pd.DataFrame(rows)
    df = df[~df['snomed_cid'].isnull()]
    entity_n = df.shape[0]
    if entity_n == 0:
        print('No entities found for MRN={}'.format(mrn))
    df.to_csv(entities_fn, index=False)
    with lock:
        mrn_counter.value += 1
        entity_counter.value += entity_n
        if mrn_counter.value % 100 == 0:
            print('Saved {} relevant entities for {} MRNs'.format(entity_counter.value, mrn_counter.value))



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

    _, _, mrns = get_mrn_status_df('valid_example')
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    with Manager() as manager:
        pool = Pool()  # By default pool will size depending on cores available
        mrn_counter = manager.Value('i', 0)
        entity_counter = manager.Value('i', 0)
        lock = manager.Lock()
        pool.map(
            partial(
                extract_entities, mrn_counter=mrn_counter, entity_counter=entity_counter, lock=lock
            ), mrns)
        pool.close()
        pool.join()

    duration(start_time)
