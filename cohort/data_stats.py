from datetime import datetime
from collections import defaultdict, Counter
from functools import partial
import os
from multiprocessing import Manager, Pool, Value
from time import time, sleep
import warnings
warnings.filterwarnings('error')

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.stats import describe
from tqdm import tqdm

notes_dir = '/nlp/projects/clinsum/notes_by_mrn'


def render_dict(d, min_val=5):
    for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True):
        if v >= min_val:
            print('{} -> {}'.format(k, str(v)))


def source_stats(mrn, num_visits=None, num_sources=None, note_types=None, note_titles=None, lock=None):
    mrn_dir = os.path.join(notes_dir, mrn)
    notes_fn = os.path.join(mrn_dir, 'notes.csv')

    try:
        notes_df = pd.read_csv(notes_fn)
    except pd.errors.EmptyDataError:
        print('Empty notes DataFrame!')
        return

    source_notes = notes_df[notes_df['is_rel_source']]
    source_notes.fillna({'title': 'N/A', 'note_type': 'N/A', 'account': 'N/A'}, inplace=True)
    n = source_notes.shape[0]

    if n == 0:
        return

    num_source = source_notes['account'].value_counts().tolist()
    num_visit = len(num_source)
    titles = source_notes['title'].tolist()
    nts = source_notes['note_type'].tolist()

    with lock:
        num_visits.append(num_visit)
        num_sources += num_source
        note_titles += titles
        note_types += nts


if __name__ == '__main__':
    mrns = os.listdir(notes_dir)
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    with Manager() as manager:
        num_sources = manager.list()
        num_visits = manager.list()
        note_types = manager.list()
        note_titles = manager.list()
        lock = manager.Lock()
        pool = Pool()  # By default pool will size depending on cores available
        pool.map(
            partial(
                source_stats, num_visits=num_visits, num_sources=num_sources, note_types=note_types,
                note_titles=note_titles, lock=lock
            ),
            mrns
        )
        pool.close()
        pool.join()

        type_cts = Counter(list(note_types))
        title_cts = Counter(list(note_titles))

        print('Source note types by count...')
        render_dict(type_cts)
        print('\n\n\n' + ('-' * 100) + '\n\n\n')
        print('Source titles by count...')
        render_dict(title_cts)
        print('\n\n\n' + ('-' * 100) + '\n\n\n')

        num_visits_stats = describe(list(num_visits))
        num_sources_stats = describe(list(num_sources))

        print('Stats for number of visits per MRN')
        print(num_visits_stats)
        print('\n\n\n' + ('-' * 100) + '\n\n\n')

        print('Stats for number of source notes per visit')
        print(num_sources_stats)

    end_time = time()
    minutes = (end_time - start_time) / 60.0
    round_factor = 0
    if minutes < 1:
        round_factor = 2
    print('Took {} minutes'.format(minutes, round(round_factor)))
