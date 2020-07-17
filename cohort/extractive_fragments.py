from collections import defaultdict, Counter
from functools import partial
import os
from multiprocessing import Manager, Pool, Value
import re
import string
from time import time

import numpy as np
import pandas as pd
from scipy.stats import describe
from tqdm import tqdm

from constants import *
from fragment_utils import Fragments
from section_utils import resolve_course, sent_toks_from_html
from utils import *

DELIM = ' ||| '


def frags(source_toks, target_toks):
    fragments = Fragments(target_toks, source_toks, convert_lower=True)
    coverage = fragments.coverage()
    density = fragments.density()
    compression = fragments.compression()
    fragment_tok_spans = list(filter(lambda x: not all([a in string.punctuation for a in x]), fragments.strings()))
    fragment_spans = list(map(lambda x: ' '.join(x).strip(), fragment_tok_spans))
    fragment_spans_str = DELIM.join(fragment_spans).strip()
    obj = {
        'coverage': coverage,
        'density': density,
        'compression': compression,
        'fragments': fragment_spans_str
    }
    return obj


def get_extractive_fragments(mrn, mrn_counter=None, lock=None):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    examples_fn = os.path.join(mrn_dir, 'examples.csv')
    examples_df = pd.read_csv(examples_fn)

    examples_df.dropna(inplace=True)
    assert examples_df.shape[0] > 0
    frag_dicts = defaultdict(list)
    for example in examples_df.to_dict('records'):
        source_toks = sent_toks_from_html(example['spacy_source_toks'])
        target_toks = sent_toks_from_html(resolve_course(example['spacy_target_toks']))
        frag_obj = frags(source_toks, target_toks)
        for k, v in frag_obj.items():
            frag_dicts[k].append(v)

    for k, v in frag_dicts.items():
        examples_df[k] = v
    examples_df.to_csv(examples_fn, index=False)

    with lock:
        mrn_counter.value += 1
        if mrn_counter.value % 1000 == 0:
            print('Processed {} MRNs'.format(mrn_counter.value))

    return frag_dicts


if __name__ == '__main__':
    _, _, mrns = get_mrn_status_df('valid_example')
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    with Manager() as manager:
        pool = Pool(processes=1)  # By default pool will size depending on cores available
        mrn_counter = manager.Value('i', 0)
        lock = manager.Lock()
        outputs = pool.map(
            partial(
                get_extractive_fragments, mrn_counter=mrn_counter, lock=lock
            ), mrns)
        pool.close()
        pool.join()

    duration(start_time)
    stats = ['compression', 'coverage', 'density']
    vals = [[], [], []]
    for output in outputs:
        for i, stat in enumerate(stats):
            vals[i] += output[stat]
    for stat, val in zip(stats, vals):
        print('Statistic={}...'.format(stat))
        print('\t', describe(val))
