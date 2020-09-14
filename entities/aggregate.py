import itertools
import os
import sys

import pandas as pd
from p_tqdm import p_uimap

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.utils import *


def generate_counts(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    entities_fn = os.path.join(mrn_dir, 'entities_relevant.csv')
    notes_fn = os.path.join(mrn_dir, 'notes.csv')

    entities_df = pd.read_csv(entities_fn)
    n_entities = entities_df.shape[0]
    assert n_entities > 0
    e_cols = set(entities_df.columns)
    notes_df = pd.read_csv(notes_fn)
    notes_df.dropna(subset=['account'], inplace=True)
    notes_df['account'] = notes_df['account'].astype('int')

    notes_df['note_id'] = notes_df['note_id'].astype('str')
    n_cols = set(notes_df.columns)
    e_cols_red = list(e_cols - n_cols) + ['note_id']

    entities_df = entities_df[e_cols_red]
    entities_df['note_id'] = entities_df['note_id'].astype('str')

    cui_map = entities_df.drop_duplicates(subset=['cui']).reset_index(drop=True).set_index(['cui']).to_dict(orient='index')
    join_df = entities_df.merge(notes_df, how='inner', on='note_id')
    join_df.dropna(subset=['account'], inplace=True)

    examples_fn = os.path.join(mrn_dir, 'examples.csv')
    valid_accounts = set(pd.read_csv(examples_fn)['account'].tolist())
    join_df = join_df[join_df['account'].isin(valid_accounts)]
    assert len(join_df) > 0

    outputs = []

    mrn_accounts = join_df['account'].unique().tolist()
    for account in mrn_accounts:
        account_df = join_df[join_df['account'] == account]
        source_df = account_df[account_df['is_source']]
        target_df = account_df[account_df['is_target']]

        source_cui_counts = source_df['cui'].value_counts().to_dict()
        target_cui_counts = target_df['cui'].value_counts().to_dict()

        all_cuis = list(set(list(source_cui_counts.keys()) + list(target_cui_counts.keys())))

        for cui in all_cuis:
            source_count = source_cui_counts.get(cui, 0)
            target_count = target_cui_counts.get(cui, 0)

            cui_info = cui_map[cui]
            outputs.append({
                'mrn': mrn,
                'account': account,
                'cui': cui,
                'cui_name': cui_info['cui_name'],
                'tui': cui_info['tui'],
                'sem_group': cui_info['sem_group'],
                'is_core': cui_info['is_core'],
                'source_count': source_count,
                'target_count': target_count,
                'definition': cui_info['definition'],
            })

    return outputs


if __name__ == '__main__':
    _, _, mrns = get_mrn_status_df('valid_example')
    outputs = list(p_uimap(generate_counts, mrns))
    outputs_flat = list(itertools.chain(*outputs))

    output_df = pd.DataFrame(outputs_flat)
    out_fn = os.path.join(out_dir, 'entity_labels.csv')
    n = len(output_df)
    output_df.sort_values(by=['source_count', 'target_count'], ascending=False, inplace=True)

    print('Saving {} CUI examples to {}'.format(n, out_fn))
    output_df.to_csv(out_fn, index=False)
