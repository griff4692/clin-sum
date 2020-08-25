from collections import Counter, defaultdict
import json
import itertools
from functools import partial
import json
import math
from multiprocessing import Pool, Manager
import os
from string import punctuation
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

from lexrank import STOPWORDS
from lexrank.algorithms.power_method import stationary_distribution
from lexrank.utils.text import tokenize
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess.constants import out_dir
from preprocess.pack_sents_lexrank import LexRank
from preprocess.section_utils import sents_from_html, pack_sentences
from preprocess.utils import get_mrn_status_df


def collect_targets(mrn):
    target_docs = []
    example_fn = os.path.join(out_dir, 'mrn', str(mrn), 'examples.csv')
    example_df = pd.read_csv(example_fn)
    for example in example_df.to_dict('records'):
        target_docs.append(' '.join(sents_from_html(example['spacy_target_toks'], convert_lower=True)))
    return target_docs


def create_lexranker():
    train_mrns = splits_df[splits_df['split'] == 'train']['mrn'].unique().tolist()
    target_docs = list(map(collect_targets, train_mrns))
    target_docs = list(itertools.chain(*target_docs))
    print('Precomputing target IDF...')
    stopwords = STOPWORDS['en']
    stopwords = stopwords.union(set([x for x in punctuation]))
    lxr = LexRank(target_docs, stopwords=stopwords)
    return lxr


if __name__ == '__main__':
    splits_df = pd.read_csv(os.path.join(out_dir, 'splits.csv'))
    all_mrns = splits_df['mrn'].unique().tolist()
    print('Building LexRank...')
    lxr = create_lexranker()
    
    obj = {
        'idf_score': dict(lxr.idf_score),
        'default': lxr.default
    }

    print('Saving IDF so we can don\'t need to recompute it...')
    out_fn = os.path.join('..', 'data', 'idf.json')
    with open(out_fn, 'w') as fd:
        json.dump(obj, fd)
