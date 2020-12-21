from collections import defaultdict
import itertools
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.utils import *


def nonzero_intersection(df, cols):
    return len(df[cols].min(axis=1).to_numpy().nonzero()[0])


if __name__ == '__main__':
    print('Loading entity labels dataframe...')
    in_fn = os.path.join(out_dir, 'entity', 'full_entities_aggregated.csv')
    df = pd.read_csv(in_fn)
    n = len(df)
    print('Finished loading. Now processing {} entities'.format(n))

    df['in_target'] = df['target_count'].apply(lambda x: 1 if x > 0 else 0)
    source_ct_df = df[['source_count', 'in_target']].groupby('source_count', as_index=False)['in_target'].mean()
    out_fn = 'data_v2/source_counts.csv'
    source_ct_df = source_ct_df.to_csv(out_fn, index=False)

    overall_source_ents = np.count_nonzero(df['source_count'])
    overall_target_ents = np.count_nonzero(df['target_count'])
    overall_intersect = nonzero_intersection(df, ['source_count', 'target_count'])

    recall = overall_intersect / float(overall_source_ents)
    precision = overall_intersect / float(overall_target_ents)
    print('Overall recall={}. Overall precision={}.'.format(recall, precision))

    # Per Visit
    account_outputs = defaultdict(list)

    accounts = df['account'].unique().tolist()
    print('{} accounts in total'.format(len(accounts)))
    account_num = len(accounts)
    for account_idx in tqdm(range(account_num)):
        account = accounts[account_idx]
        account_df = df[df['account'] == account]
        num_source_ents = np.count_nonzero(account_df['source_count'])
        num_target_ents = np.count_nonzero(account_df['target_count'])
        num_intersect = nonzero_intersection(account_df, ['source_count', 'target_count'])

        recall = num_intersect / max(1.0, float(num_source_ents))
        precision = num_intersect / max(1.0, float(num_target_ents))

        account_outputs['mrn'].append(account_df['mrn'].iloc[0])
        account_outputs['account'].append(account)
        account_outputs['recall'].append(recall)
        account_outputs['precision'].append(precision)
        account_outputs['source_ent_count'].append(num_source_ents)
        account_outputs['target_ent_count'].append(num_target_ents)

    account_df = pd.DataFrame(account_outputs)
    out_fn = 'data_v2/account_stats.csv'
    account_df.to_csv(out_fn, index=False)

    cuis = df['cui'].unique().tolist()
    cui_outputs = defaultdict(list)
    cui_num = len(cuis)
    for cui_idx in tqdm(range(cui_num)):
        cui = cuis[cui_idx]
        cui_df = df[df['cui'] == cui]

        account_tf = len(cui_df['account'].unique())

        cui_outputs['cui'].append(cui)
        cui_outputs['tui'].append(cui_df['tui'].iloc[0])
        cui_outputs['sem_group'].append(cui_df['sem_group'].iloc[0])
        cui_outputs['account_tf'].append(account_tf)
        cui_outputs['account_tf_norm'].append(account_tf / float(account_num))

        num_source_ents = np.count_nonzero(cui_df['source_count'])
        num_target_ents = np.count_nonzero(cui_df['target_count'])
        num_intersect = nonzero_intersection(cui_df, ['source_count', 'target_count'])

        recall = num_intersect / max(1.0, float(num_source_ents))
        precision = num_intersect / max(1.0, float(num_target_ents))

        cui_outputs['recall'].append(recall)
        cui_outputs['precision'].append(precision)
        cui_outputs['source_ent_count'].append(num_source_ents)
        cui_outputs['target_ent_count'].append(num_target_ents)

    cui_df = pd.DataFrame(cui_outputs)
    out_fn = 'data/cui_stats.csv'
    cui_df.to_csv(out_fn, index=False)

    tuis = cui_df['tui'].unique().tolist()
    tui_output = defaultdict(list)
    for tui in tuis:
        tui_df = cui_df[cui_df['tui'] == tui]
        avg_recall = tui_df['recall'].mean()
        avg_precision = tui_df['precision'].mean()
        tui_output['tui'].append(tui)
        tui_output['recall'].append(avg_recall)
        tui_output['precision'].append(avg_precision)
        tui_output['n'].append(len(tui_df))

    tui_df = pd.DataFrame(tui_output)
    out_fn = 'data_v2/tui_stats.csv'
    tui_df.to_csv(out_fn, index=False)

    sem_groups = cui_df['sem_group'].unique().tolist()
    sem_group_output = defaultdict(list)
    for sem_group in sem_groups:
        sem_group_df = cui_df[cui_df['sem_group'] == sem_group]
        avg_recall = sem_group_df['recall'].mean()
        avg_precision = sem_group_df['precision'].mean()
        sem_group_output['sem_group'].append(sem_group)
        sem_group_output['recall'].append(avg_recall)
        sem_group_output['precision'].append(avg_precision)
        sem_group_output['n'].append(len(sem_group_df))

    sem_group_df = pd.DataFrame(sem_group_output)
    out_fn = 'data_v2/sem_group_stats.csv'
    sem_group_df.to_csv(out_fn, index=False)
