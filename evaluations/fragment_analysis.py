import itertools
import os
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import argparse
import numpy as np
import pandas as pd
from p_tqdm import p_uimap

from preprocess.constants import out_dir
from preprocess.extractive_fragments import DELIM as FRAG_DELIM
from preprocess.section_utils import resolve_course, sents_from_html, sent_toks_from_html
from preprocess.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to measure fragment length against positional rank')
    parser.add_argument('--max_n', default=-1, type=int)

    args = parser.parse_args()

    validation_df = get_records(split='validation')
    validation_records = validation_df.to_dict('records')

    ranks = [[] for _ in range(10)]

    for record in validation_records:
        frags = [] if type(record['fragments']) == float else record['fragments'].split(FRAG_DELIM)
        for frag_idx, frag in enumerate(frags):
            frag_idx_adj = min(9, frag_idx)
            frag_len = len(frag.split(' '))
            ranks[frag_idx_adj].append(frag_len)

    for rank in range(len(ranks)):
        print(rank, ',',  np.array(ranks[rank]).mean())
