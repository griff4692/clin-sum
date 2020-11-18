import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *


if __name__ == '__main__':
    volunteers_df = pd.read_csv('data/volunteers.csv')

    in_fn = os.path.join(out_dir, 'human', 'form_data_flat.csv')
    df = pd.read_csv(in_fn)
    n = len(df)
    assert n == 100
    questions_fn = 'data/questions.txt'
    with open(questions_fn, 'r') as fd:
        questions = list(filter(lambda x: len(x) > 0, list(map(lambda x: x.strip(), fd.readlines()))))

    df['assignment'] = ['n/a' for _ in range(n)]
    for volunteer_obj in volunteers_df.to_dict('records'):
        ct, name = volunteer_obj['assignments'], volunteer_obj['alias']
        open_df = df[df['assignment'] == 'n/a']
        account_counts = dict(open_df['visit_id'].value_counts())
        double_accounts = []
        single_accounts = []
        for account, ct in account_counts.items():
            if ct == 1:
                single_accounts.append(account)
            else:
                double_accounts.append(account)
        free_n = min(assignment_ct, len(double_accounts))
        sample_accounts = list(np.random.choice(double_accounts, size=free_n, replace=False))
        if free_n < assignment_ct:
            extra = assignment_ct - free_n
            sample_accounts += list(np.random.choice(single_accounts, size=extra, replace=False))
        for account in sample_accounts:
            sample_df = open_df[open_df['visit_id'] == account].sample(n=1)
            df.loc[sample_df.index, 'assignment'] = name
    out_fn = os.path.join(out_dir, 'human', 'target_form.csv')

    for i, question in enumerate(questions):
        df[question] = ['' for _ in range(n)]
        df['comments_{}'.format(i + 1)] = ['' for _ in range(n)]
    df.sort_values(by='assignment', inplace=True)
    df = df[df['assignment'] != 'n/a']
    df.to_csv(out_fn, index=False)
    visit_ids = df['visit_id'].tolist()
    mrn_dir = os.path.join(out_dir, 'mrn')
    mrns_accounts = df[['mrn', 'visit_id']].drop_duplicates().to_records(index=False)
    source_data = []
    for mrn, account in mrns_accounts:
        notes_fn = os.path.join(mrn_dir, str(mrn), 'notes.csv')
        notes_df = pd.read_csv(notes_fn)
        account_df = notes_df[notes_df['account'] == account]
        source_notes = account_df[account_df['is_source']]
        assert len(source_notes) > 0
        for record in source_notes.to_dict('records'):
            t = record['text'].replace('\u00A0', ' ')
            t = re.sub(r'\n+', ' <newline> ', t).strip()
            t = re.sub(r'\s+', ' ', t)
            source_data.append({
                'visit_id': account,
                'mrn': mrn,
                'timestamp': record['timestamp'],
                'med_code': record['med_code'],
                'note_type': record['note_type'],
                'title': record['title'],
                'text': t.strip()
            })

    source_data = pd.DataFrame(source_data)
    source_data.sort_values(by=['mrn', 'timestamp'], inplace=True)
    out_fn = os.path.join(out_dir, 'human', 'source_form.csv')
    source_data = source_data[source_data['visit_id'].isin(visit_ids)]
    source_data.to_csv(out_fn, index=False)
