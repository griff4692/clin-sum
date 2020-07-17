from collections import defaultdict, Counter
import itertools
from multiprocessing import Manager, Pool, Value
import os
import pickle
import re
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from constants import *
from utils import *


def determine_split(mrn, train_mrns, val_mrns, test_mrns):
    if mrn in train_mrns:
        return 'train'
    elif mrn in val_mrns:
        return 'validation'
    elif mrn in test_mrns:
        return 'test'
    else:
        raise Exception('Phantom MRN={} not found in train, validation, or test sets'.format(mrn))


def collect_valid_accounts(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    examples_fn = os.path.join(mrn_dir, 'examples.csv')
    examples_df = pd.read_csv(examples_fn)
    return examples_df[['mrn', 'account']].to_dict('records')


def disjoint(a, b):
    return len(a.intersection(b)) == 0


def validate_splits(splits):
    if not (disjoint(splits['train'], splits['validation']) and disjoint(splits['train'], splits['test'])
            and disjoint(splits['validation'], splits['test'])):
        raise Exception('Train-Validation-Test set MRN leakage.  Please debug!')


if __name__ == '__main__':
    _, _, mrns = get_mrn_status_df('valid_example')
    with open(hiv_mrn_fn, 'rb') as fd:
        hiv_mrns = set(pickle.load(fd)['mrn'].unique().tolist())

    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    with Manager() as manager:
        pool = Pool()  # By default pool will size depending on cores available
        mrn_counter = manager.Value('i', 0)
        lock = manager.Lock()
        output = pool.map(collect_valid_accounts, mrns)
        pool.close()
        pool.join()

    duration(start_time)
    print('Computing train/val/test splits...')
    train_mrns, other = train_test_split(mrns, train_size=0.7)
    val_mrns, test_mrns = train_test_split(other, train_size=0.5)

    splits = {
        'train': set(train_mrns),
        'validation': set(val_mrns),
        'test': set(test_mrns)
    }

    print('Validating splits for leakage.  Will raise an exception if any issues...')
    validate_splits(splits)

    print('Assigning specific visits to train-test-validation based on MRN splits...')
    flattened_output = list(itertools.chain(*output))
    df = pd.DataFrame(flattened_output)
    df['mrn'] = df['mrn'].astype('str')
    df['split'] = df['mrn'].apply(lambda mrn: determine_split(mrn, train_mrns, val_mrns, test_mrns))
    df['hiv'] = df['mrn'].apply(lambda mrn: mrn in hiv_mrns)

    for type in ['train', 'validation', 'test']:
        n_mrn = len(splits[type])
        n_ex = len(df[df['split'] == type])
        print('{} set: {} mrns across {} visits'.format(type, n_mrn, n_ex))

    out_fn = os.path.join(out_dir, 'splits.csv')
    print('Saving splits to {}'.format(out_fn))
    df.to_csv(out_fn, index=False)
