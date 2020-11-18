from datetime import datetime
from collections import defaultdict, Counter
import json
from functools import partial
import itertools
import os
import re
import sys
from time import time

import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from p_tqdm import p_uimap
import spacy
import scispacy
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.section_utils import paragraph_from_html
from preprocess.tokenize_mrns import sent_segment
from preprocess.utils import *


def get_grid(record):
    mrn = record['mrn']
    account = record['account']
    target_paragraphs = [sent_segment(t, sentencizer=sentencizer) for t in paragraph_from_html(record['target_str'])]
    target_sents = list(itertools.chain(*target_paragraphs))
    ent_fn = os.path.join(out_dir, 'entity', 'entities', '{}_{}.csv'.format(mrn, account))
    ent_df = pd.read_csv(ent_fn)
    ent_df.dropna(subset=['cui', 'cui_name', 'source_value'], inplace=True)
    ent_target_df = ent_df[ent_df['is_target']]
    egrid = defaultdict(list)
    ent_target_df['source_lens'] = ent_target_df['source_value'].apply(lambda x: len(x))
    ent_target_df_records = ent_target_df.sort_values('source_lens', ascending=False).to_dict('records')
    num_target_sents = len(target_sents)
    for entity in ent_target_df_records:
        source_val = entity['source_value']
        search_regex = re.escape(source_val)
        cui = entity['cui']
        for sidx, sent in enumerate(target_sents):
            match = re.search(search_regex, sent)
            if match is not None:
                s, e = match.start(), match.end()
                egrid[cui].append({
                    'sent_idx': sidx,
                    'start_idx': s,
                    'end_idx': e,
                    'source_val': source_val,
                    'tui': entity['tui'],
                    'sem_group': entity['sem_group']
                })

    return {
        'egrid': egrid, 'mrn': mrn, 'account': account, 'split': record['split'], 'hiv': record['hiv'],
        'num_target_sents': num_target_sents
    }


if __name__ == '__main__':
    print('Loading Spacy...')
    sentencizer = spacy.load('en_core_sci_lg', disable=['tagger', 'parser', 'ner', 'textcat'])
    sentencizer.add_pipe(sentencizer.create_pipe('sentencizer'))

    print('Getting records')
    splits = ['validation', 'train']
    examples = get_records(split=splits, mini=args.mini).to_dict('records')
    n = len(examples)

    egrids = list(p_uimap(get_grid, examples, num_cpus=0.8))
    out_fn = os.path.join(out_dir, 'egrids.json')
    print('Done! Now saving {} e-grids to {}'.format(len(egrids), out_fn))
    with open(out_fn, 'w') as fd:
        json.dump(egrids, fd)

    egrids_small = list(np.random.choice(x, size=1000, replace=False))
    out_fn = os.path.join(out_dir, 'egrids_small.json')
    with open(out_fn, 'w') as fd:
        json.dump(egrids_small, fd)
