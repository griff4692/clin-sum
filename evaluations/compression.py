import os
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from cohort.constants import *


MAX_N = 10000


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compression Evaluation Script')
    parser.add_argument('--split', default='validation')

    args = parser.parse_args()

    df = pd.read_csv(os.path.join(out_dir, 'splits.csv'))
    val_mrns = df[df['split'] == args.split]['mrn'].unique().tolist()

    plt.style.use('seaborn-deep')
    src_tok_lens = []
    target_tok_lens = []
    n = len(val_mrns)
    for i in tqdm(range(n)):
        df = pd.read_csv(os.path.join(out_dir, 'mrn', str(val_mrns[i]), 'examples.csv'))
        source_toks = df['spacy_source_tok_ct'].tolist()
        target_toks = df['spacy_target_tok_ct'].tolist()
        for st, tt in zip(source_toks, target_toks):
            if st < MAX_N and tt < MAX_N:
                src_tok_lens.append(st)
                target_tok_lens.append(tt)

    plt.hist([src_tok_lens, target_tok_lens], bins=1000, label=['Source Lengths', 'Target Lengths'])
    plt.legend(loc='upper right')
    plt.savefig('src_target_dist.png')
    plt.close()

    compression_ratio = [x[0] / float(x[1]) for x in zip(target_tok_lens, src_tok_lens)]
    plt.hist([compression_ratio], bins=1000, label=['Compression Ratio'])
    plt.xlim([0, 1])
    plt.savefig('compression_ratio.png')
    plt.close()

    plt.scatter(x=src_tok_lens, y=target_tok_lens)
    plt.xlabel('Source Lengths')
    plt.ylabel('Target Lengths')
    plt.savefig('src_target_scatter.png')
    plt.close()
