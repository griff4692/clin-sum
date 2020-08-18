import os

import argparse
import numpy as np
import pandas as pd

from constants import *
from section_utils import sents_from_html
from utils import *


def sent_sample(record):
    target_sents = sents_from_html(record['spacy_target_toks'])
    rand_sent = np.random.choice(target_sents, size=1)[0]
    print('Source:')
    print(record['spacy_source_toks'])
    print('\nRandomly sampled sentence from target:')
    print(rand_sent)
    print('\n\n\n\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate and visualize (valid) examples.')

    parser.add_argument('--mode', default='generate', choices=['generate', 'visualize'])
    parser.add_argument('--idx', default=0, type=int)
    args = parser.parse_args()

    if args.mode == 'generate':
        _, _, mrns = get_mrn_status_df('valid_example')
        n = len(mrns)
        sample_n = min(n, 25)
        mrn_sample = np.random.choice(mrns, size=sample_n, replace=False)
        dfs = []
        for mrn in mrn_sample:
            mrn_dir = os.path.join(out_dir, 'mrn', str(mrn))
            examples_df = pd.read_csv(os.path.join(mrn_dir, 'examples.csv'))
            dfs.append(examples_df)

        dfs = pd.concat(dfs)
        out_fn = os.path.join(out_dir, 'sample_examples.csv')
        dfs.to_csv(out_fn, index=False)
    elif args.mode == 'visualize':
        in_fn = os.path.join(out_dir, 'sample_examples.csv')
        df = pd.read_csv(in_fn)
        records = df.to_dict('records')
        sent_sample(records[args.idx])
    else:
        raise Exception('Not going to happen!')