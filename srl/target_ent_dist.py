from collections import defaultdict
import json
import math
import os
import sys

import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.utils import *
from preprocess.section_utils import *
from preprocess.tokenize_mrns import strip_punc


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to measure distribution of CUIs.')
    parser.add_argument('-mini', default=False, action='store_true')
    parser.add_argument('--max_n', type=int, default=None)
    parser.add_argument('-analyze_source', default=False, action='store_true')

    args = parser.parse_args()
    mini_str = '_small' if args.mini else ''
    base_dir = os.path.join(out_dir, 'entity')
    cui_tui_group_fn = os.path.join(base_dir, 'cui_tui_group.json')
    if os.path.exists(cui_tui_group_fn):
        with open(cui_tui_group_fn, 'r') as fd:
            cui_tui_group_map = json.load(fd)
    else:
        in_ent_fn = os.path.join(base_dir, 'full_entities.csv')
        ent_df = pd.read_csv(in_ent_fn)
        ent_df.drop_duplicates(subset=['cui'], inplace=True)
        cui_tui_group_map = dict(zip(ent_df['cui'], zip(ent_df['tui'], ent_df['sem_group'])))
        with open(cui_tui_group_fn, 'w') as fd:
            json.dump(cui_tui_group_map, fd)

    in_fn = os.path.join(out_dir, 'srl_packed_examples{}.csv'.format(mini_str))
    if args.max_n is not None:
        in_fn = os.path.join(out_dir, 'srl_packed_examples_{}.csv'.format(str(args.max_n)))
    transition_mat = np.zeros([3, 3])
    sem_group_set = ['Chemicals & Drugs', 'Disorders', 'Procedures']

    print('Loading {}...'.format(in_fn))
    df = pd.read_csv(in_fn)
    srl_col = 'srl_packed_source' if args.analyze_source else 'srl_packed_target'
    df.dropna(subset=[srl_col], inplace=True)
    dist = []
    records = df.to_dict('records')
    n = len(records)
    for record_idx in tqdm(range(n)):
        record = records[record_idx]
        all_text = record[srl_col]
        texts = all_text.split('<d>')
        for text in texts:
            cuis = re.findall(r'cui=(\w+) ', text)
            n = len(cuis)
            prev_idx = -1
            record_dist = []
            for idx in range(len(cuis)):
                cui = cuis[idx]
                if cui in cui_tui_group_map:
                    tui, sem_group = cui_tui_group_map[cui]
                else:
                    tui, sem_group = None, None
                    print('Unrecognized CUI={}'.format(cui))

                record_dist.append({
                    'cui': cui,
                    'tui': tui,
                    'sem_group': sem_group,
                    'ent_pos_start': idx,
                    'total_tok_len': n
                })

            for i in range(1, len(record_dist)):
                row_sg = record_dist[i - 1]['sem_group']
                col_sg = record_dist[i]['sem_group']
                if row_sg in sem_group_set and col_sg in sem_group_set:
                    row = sem_group_set.index(row_sg)
                    col = sem_group_set.index(col_sg)
                    transition_mat[row, col] += 1
            dist += record_dist

    dist_df = pd.DataFrame(dist)
    target_n = min(100000, len(dist_df))
    print('Sampling down from {} to {}'.format(len(dist_df), target_n))
    dist_df = dist_df.sample(n=target_n, replace=False)
    dist_df['tentile'] = dist_df['ent_pos_start'].combine(
        dist_df['total_tok_len'], lambda a, b: int((float(a) / float(b)) // 0.1) + 1)
    sem_group_counts = dist_df.groupby(['sem_group', 'tentile']).size().reset_index(name='freq')
    sem_group_counts = sem_group_counts[sem_group_counts['sem_group'].isin(sem_group_set)]
    for sem_group in sem_group_counts['sem_group'].unique():
        sub_df = sem_group_counts[sem_group_counts['sem_group'] == sem_group]
        total = sub_df['freq'].sum()
        print('Sem Group={}. Total count={}'.format(sem_group, total))
        for record in sub_df.to_dict('records'):
            print(record['tentile'], ',', record['freq'] / float(total))
        print('\n')
    target_str = '_source' if args.analyze_source else '_target'
    out_fn = os.path.join(base_dir, 'target_ent_distribution_sample{}.csv'.format(target_str))
    print('Saving {} rows to {}'.format(len(dist_df), out_fn))
    dist_df.to_csv(out_fn, index=False)

    print('Transition Matrix:')
    print('\t'.join(sem_group_set))
    for r in range(len(sem_group_set)):
        str_row = '\t'.join([str(x) for x in transition_mat[r]])
        print(sem_group_set[r] + '\t' + str_row)
