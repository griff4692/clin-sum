from collections import defaultdict, Counter
from functools import partial
import os
import re
import string
from time import time

import numpy as np
import pandas as pd
from p_tqdm import p_uimap
from scipy.stats import describe

from constants import *
from fragment_utils import Fragments
from preprocess.utils import *
from section_utils import resolve_course, sent_toks_from_html

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


def get_extractive_fragments(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    examples_fn = os.path.join(mrn_dir, 'examples.csv')
    examples_df = pd.read_csv(examples_fn)

    examples_df.dropna(subset=['spacy_source_toks', 'spacy_target_toks'], inplace=True)
    frag_dicts = defaultdict(list)
    if examples_df.shape[0] == 0:
        print('MRN={} has no valid tokens'.format(mrn))
        return frag_dicts
    for example in examples_df.to_dict('records'):
        source_toks = sent_toks_from_html(example['spacy_source_toks'])
        target_toks = sent_toks_from_html(resolve_course(example['spacy_target_toks']))
        frag_obj = frags(source_toks, target_toks)
        for k, v in frag_obj.items():
            frag_dicts[k].append(v)

    for k, v in frag_dicts.items():
        examples_df[k] = v
    examples_df.to_csv(examples_fn, index=False)

    return frag_dicts


if __name__ == '__main__':
    _, _, mrns = get_mrn_status_df('valid_example')
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    outputs = list(p_uimap(get_extractive_fragments, mrns))
    duration(start_time)
    stat_names = ['compression', 'coverage', 'density']
    stats = defaultdict(list)
    for output in outputs:
        for stat in stat_names:
            stats[stat] += output[stat]

    df = pd.DataFrame(stats)
    out_fn = '../evaluations/results/extractiveness.csv'
    df.to_csv(out_fn, index=False)

    for stat, vals in stats.items():
        print('Statistic={}...'.format(stat))
        print('\t', describe(vals))
