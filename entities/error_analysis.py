from collections import Counter
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import out_dir


def add_token_data(df):
    source_toks = []
    target_toks = []

    for record in df.to_dict('records'):
        mrn = record['mrn']
        account = record['account']

        example_fn = os.path.join(out_dir, 'mrn', str(mrn), 'examples.csv')
        example_df = pd.read_csv(example_fn)

        account_df = example_df[example_df['account'] == account]
        assert len(account_df) == 1

        source_toks.append(account_df.iloc[0]['spacy_source_toks'])
        target_toks.append(account_df.iloc[0]['spacy_target_toks'])

    df['source_toks'] = source_toks
    df['target_toks'] = target_toks


def sample(df, ct):
    rows = []
    cui_samples = np.random.choice(df['cui'].unique().tolist(), size=ct, replace=True)
    cui_sample_cts = Counter(cui_samples)
    for cui, k in cui_sample_cts.items():
        cui_df = df[df['cui'] == cui]
        rows.append(cui_df.sample(n=min(k, len(cui_df)), replace=False))
    return pd.concat(rows)


if __name__ == '__main__':
    ent_fn = os.path.join(out_dir, 'entity_labels_50ksample.csv')
    ent_df = pd.read_csv(ent_fn)
    ent_df = ent_df[ent_df['is_core']]
    ent_df = ent_df[ent_df['sem_group'] == 'Disorders']

    print('Getting samples of in source, not in target')
    only_in_source_df = ent_df[ent_df['target_count'] == 0]
    only_in_source_sample_df = sample(only_in_source_df, 25)
    add_token_data(only_in_source_sample_df)
    only_in_source_sample_fn = os.path.join('data', 'only_in_source_examples.csv')
    only_in_source_sample_df.to_csv(only_in_source_sample_fn, index=False)

    print('Getting samples of in target, not in source')
    only_in_target_df = ent_df[ent_df['source_count'] == 0]
    only_in_target_sample_df = sample(only_in_target_df, 25)
    add_token_data(only_in_target_sample_df)
    only_in_target_sample_fn = os.path.join('data', 'only_in_target_examples.csv')
    only_in_target_sample_df.to_csv(only_in_target_sample_fn, index=False)
