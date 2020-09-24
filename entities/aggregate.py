import itertools
import os
import sys

import pandas as pd
from p_tqdm import p_uimap

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.utils import *


def generate_counts(account):
    outputs = []

    account_df = ent_df[ent_df['account'] == account]
    mrn = account_df['mrn'].iloc[0]
    source_df = account_df[account_df['is_source']]
    target_df = account_df[account_df['is_target']]

    source_cui_counts = source_df['cui'].value_counts().to_dict()
    target_cui_counts = target_df['cui'].value_counts().to_dict()

    all_cuis = list(set(list(source_cui_counts.keys()) + list(target_cui_counts.keys())))
    for cui in all_cuis:
        source_count = source_cui_counts.get(cui, 0)
        target_count = target_cui_counts.get(cui, 0)
        outputs.append({
            'mrn': mrn,
            'account': account,
            'cui': cui,
            'source_count': source_count,
            'target_count': target_count,
        })

    return outputs


if __name__ == '__main__':
    ent_fn = os.path.join(out_dir, 'entity', 'full_entities.csv')
    ent_df = pd.read_csv(ent_fn)
    accounts = ent_df['account'].unique().tolist()
    num_mrns = len(ent_df['mrn'].unique())
    print('Collected {} entities for {} unique visits across {} patients'.format(len(ent_df), len(accounts), num_mrns))
    outputs = list(p_uimap(generate_counts, accounts, num_cpus=0.5))
    outputs_flat = list(itertools.chain(*outputs))

    output_df = pd.DataFrame(outputs_flat)
    out_fn = os.path.join(out_dir, 'entity', 'full_entities_aggregated.csv')
    n = len(output_df)
    output_df.sort_values(by=['source_count', 'target_count'], ascending=False, inplace=True)

    print('Saving {} CUI examples to {}'.format(n, out_fn))
    output_df.to_csv(out_fn, index=False)
