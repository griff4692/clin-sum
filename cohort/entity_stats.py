from datetime import datetime
from collections import defaultdict, Counter
from functools import partial
import os
from multiprocessing import Manager, Pool, Value
from time import time
import warnings

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


def calculate_pr(mrn, mrn_ct=None, not_ready_ct=None, recalls=None, precisions=None, lock=None):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    entities_fn = os.path.join(mrn_dir,  'entities.csv')
    notes_fn = os.path.join(mrn_dir, 'notes.csv')

    if not os.path.exists(entities_fn):
        with lock:
            not_ready_ct.value += 1
        return None

    entities_df = pd.read_csv(entities_fn)
    n_entities = entities_df.shape[0]
    assert n_entities > 0
    e_cols = set(entities_df.columns)
    notes_df = pd.read_csv(notes_fn)

    notes_df['note_id'] = notes_df['note_id'].astype('str')
    n_cols = set(notes_df.columns)
    e_cols_red = list(e_cols - n_cols) + ['note_id']

    entities_df = entities_df[e_cols_red]
    entities_df['note_id'] = entities_df['note_id'].astype('str')
    join_df = entities_df.merge(notes_df, how='inner', on='note_id')
    accounts = join_df['account'].unique().tolist()
    cui_type_map = join_df.set_index('cui')['umls_type'].to_dict()
    types = defaultdict(list)
    rs, ps = [], []
    for account in accounts:
        account_df = join_df[join_df['account'] == account]

        source_df = account_df[account_df['is_source']]
        target_df = account_df[account_df['is_target']]
        target_df['len'] = target_df['text'].apply(lambda x: len(x))
        max_len = target_df['len'].max()
        target_df = target_df[target_df['len'] == max_len]

        source_df.drop_duplicates(subset=['cui'], inplace=True)
        target_df.drop_duplicates(subset=['cui'], inplace=True)

        source_cuis = set(source_df['cui'].tolist())
        target_cuis = set(target_df['cui'].tolist())
        cui_overlap = source_cuis.intersection(target_cuis)
        just_source = source_cuis - target_cuis
        just_target = target_cuis - source_cuis

        overlap_types = list(map(lambda x: cui_type_map[x], list(cui_overlap)))
        source_only_types = list(map(lambda x: cui_type_map[x], list(just_source)))
        target_only_types = list(map(lambda x: cui_type_map[x], list(just_target)))
        source_types = list(map(lambda x: cui_type_map[x], list(source_cuis)))
        target_types = list(map(lambda x: cui_type_map[x], list(target_cuis)))

        types['overlap'] += overlap_types
        types['source_only'] += source_only_types
        types['target_only'] += target_only_types
        types['source'] += source_types
        types['target'] += target_types

        num = float(len(cui_overlap))
        p = num / float(max(1.0, len(target_cuis)))
        r = num / float(max(1.0, len(source_cuis)))

        rs.append(r)
        ps.append(p)

    with lock:
        recalls += rs
        precisions += ps
        mrn_ct.value += 1
        if mrn_ct.value % 1000 == 0:
            r_mean = sum(recalls) / float(len(recalls))
            p_mean = sum(precisions) / float(len(precisions))
            print('Processed {} MRNs ({} not ready). Recall={}. Precision={} '.format(
                mrn_ct.value, not_ready_ct.value, r_mean, p_mean))
    return types


if __name__ == '__main__':
    _, _, mrns = get_mrn_status_df('valid_example')
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    with Manager() as manager:
        pool = Pool()  # By default pool will size depending on cores available
        lock = manager.Lock()
        recalls = manager.list()
        precisions = manager.list()
        mrn_ct = manager.Value('i', 0)
        not_ready_ct = manager.Value('i', 0)
        type_counts = list(pool.map(
            partial(
                calculate_pr, mrn_ct=mrn_ct, not_ready_ct=not_ready_ct, recalls=recalls,
                precisions=precisions, lock=lock
            ), mrns))
        pool.close()
        pool.join()

        print('Recall stats...')
        print(describe(recalls))

        print('Precision stats...')
        print(describe(precisions))

    overlap_types = []
    source_only_types = []
    target_only_types = []
    source_types = []
    target_types = []
    for tc in type_counts:
        if tc is None:
            continue
        overlap_types += tc['overlap']
        source_only_types += tc['source_only']
        target_only_types += tc['target_only']
        source_types += tc['source']
        target_types += tc['target']

    duration(start_time)

    overlap_type_counts = ('overlap', Counter(overlap_types))
    source_only_type_counts = ('source_only', Counter(source_only_types))
    target_only_type_counts = ('target_only', Counter(target_only_types))
    target_type_counts = ('target', Counter(target_types))
    source_type_counts = ('source', Counter(source_types))

    d = [
        overlap_type_counts,
        source_only_type_counts,
        target_only_type_counts,
        target_type_counts,
        source_type_counts
    ]

    df = []
    cols = ['name', 'umls_type', 'count', 'perc']
    for name, counts in d:
        denom = sum(counts.values())
        for k, v in counts.items():
            df.append((
                name,
                k,
                int(v),
                float(v) / max(1.0, float(denom))
            ))
    df = pd.DataFrame(df, columns=cols)
    df.sort_values(by=['name', 'count'], ascending=False, inplace=True)
    df.to_csv(os.path.join(out_dir, 'umls_type_counts.csv'), index=False)
