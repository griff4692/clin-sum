from collections import defaultdict, Counter
import itertools
from multiprocessing import Manager, Pool, Value
import os
import pickle
import re
import sys
from time import time

import numpy as np
import pandas as pd
from p_tqdm import p_imap
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.utils import *


def determine_split(mrn, train_mrns, val_mrns, test_mrns):
    if mrn in train_mrns:
        return 'train'
    elif mrn in val_mrns:
        return 'validation'
    elif mrn in test_mrns:
        return 'test'
    else:
        raise Exception('Phantom MRN={} not found in train, validation, or test sets'.format(mrn))


def disjoint(a, b):
    return len(a.intersection(b)) == 0


def validate_splits(splits):
    if not (disjoint(splits['train'], splits['validation']) and disjoint(splits['train'], splits['test'])
            and disjoint(splits['validation'], splits['test'])):
        raise Exception('Train-Validation-Test set MRN leakage.  Please debug!')


if __name__ == '__main__':
    full_example_fn = os.path.join(out_dir, 'full_examples.csv')
    print('Loading full examples...')
    df = pd.read_csv(full_example_fn)

    mrns = list(df['mrn'].unique())
    mrn_n = len(mrns)
    print('Loaded {} examples for {} mrns'.format(len(df), mrn_n))

    print('Getting HIV patients...')
    with open(hiv_mrn_fn, 'rb') as fd:
        hiv_mrns = set(pickle.load(fd)['mrn'].unique().tolist())

    print('Computing train/val/test splits...')
    train_mrns, other = train_test_split(mrns, train_size=0.9)
    val_mrns, test_mrns = train_test_split(other, train_size=0.25)
    splits = {
        'train': set(train_mrns),
        'validation': set(val_mrns),
        'test': set(test_mrns)
    }

    print('Validating splits for leakage.  Will raise an exception if any issues...')
    validate_splits(splits)

    print('Assigning specific visits to train-test-validation based on MRN splits...')
    df['split'] = df['mrn'].apply(lambda mrn: determine_split(mrn, train_mrns, val_mrns, test_mrns))
    df['hiv'] = df['mrn'].apply(lambda mrn: mrn in hiv_mrns)

    for type in ['train', 'validation', 'test']:
        n_mrn = len(splits[type])
        n_ex = len(df[df['split'] == type])
        print('{} set: {} mrns across {} visits'.format(type, n_mrn, n_ex))

    print('Saving splits to {}'.format(full_example_fn))
    df.to_csv(full_example_fn, index=False)
