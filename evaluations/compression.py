import os
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess.constants import *
from preprocess.section_utils import *
from preprocess.extractive_fragments import DELIM as FRAG_DELIM


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compression Evaluation Script')
    args = parser.parse_args()
    all_frag_lens = []
    all_frag_stds = []
    validation_records = get_records(split='validation')
    for frag_set in validation_records['fragments'].dropna().tolist():
        frags = frag_set.split(FRAG_DELIM)
        frag_lens = [len(f.split(' ')) for f in frags]
        all_frag_lens += frag_lens
        all_frag_stds.append(np.std(frag_lens))

    print(np.mean(all_frag_stds))
    with open('tmp.txt', 'w') as fd:
        fd.write('\n'.join(map(str, all_frag_lens)))
