from collections import Counter, defaultdict
import json
import os
from string import punctuation
import sys

import argparse
from fuzzysearch import find_near_matches
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from p_tqdm import p_uimap
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.extractive_fragments import DELIM as FRAG_DELIM
from preprocess.section_utils import resolve_course, sent_toks_from_html
from preprocess.tokenize_mrns import sent_segment
from preprocess.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to build entity-specific LM.')

    args = parser.parse_args()
    splits = ['validation', 'train']
    examples = get_records(split=splits, mini=False).to_dict('records')
    n = len(examples)

    target_vocabulary = set()
    account_wcs = {}
    for example in tqdm(examples):
        account = example['account']
        mrn = example['mrn']
        toks = sent_toks_from_html(example['spacy_target_toks'], convert_lower=True)
        toks_lower = [t.lower() for t in toks]
        for tl in toks_lower:
            target_vocabulary.add(tl)
        key = '{}_{}'.format(mrn, account)
        account_wcs[key] = Counter(toks_lower)

    # Get disorder entities - we just need for each account, the set of target CUIs
    cui_wcs = defaultdict(Counter)

    ents_fn = os.path.join(out_dir, 'entity', 'full_entities_aggregated.csv')
    ents_df = pd.read_csv(ents_fn)
    # Target only
    target_ents_df = ents_df[ents_df['target_count'] > 0]
    keys = target_ents_df['mrn'].combine(target_ents_df['account'], lambda a, b: '{}_{}'.format(a, b))

    cui_keys = list(zip(target_ents_df['cui'].tolist(), keys.tolist()))
    for cui, key in tqdm(cui_keys):
        cui_wcs[cui].update(account_wcs[key])

    out_fn = os.path.join(out_dir, 'entity', 'cui_tok_freqs.json')
    print('Saving vocabulary counts for {} entities to {}'.format(len(cui_wcs), out_fn))
    with open(out_fn, 'w') as fd:
        json.dump(cui_wcs, fd)
