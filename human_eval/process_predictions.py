import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from evaluations.rouge import compute, prepare_str_for_rouge


if __name__ == '__main__':
    in_fn = os.path.join(out_dir, 'predictions', 'oracle_greedy_rel_hiv.csv')
    print('Loading predictions from {}...'.format(in_fn))
    prediction_df = pd.read_csv(in_fn)
    prediction_df['prediction'] = prediction_df['prediction'].fillna(value='')

    visit_fn = os.path.join(out_dir, 'visits.csv')
    visits_df = pd.read_csv(visit_fn)[['mrn', 'account', 'admit_date_min', 'discharge_date_max']]
    df = prediction_df.merge(visits_df, on=['mrn', 'account'], how='inner')

    # Remove 2010 as examples (missing data for that year)
    df = df[df['admit_date_min'].apply(lambda x: int(x[:4]) > 2010)]

    # Randomly drop duplicate mrns
    df = df.sample(frac=1).reset_index(drop=True)
    df.drop_duplicates(subset=['mrn'], keep='first', inplace=True)

    median_sum_len = df['ref_len'].median()
    df['abs_distance'] = df['ref_len'].apply(lambda x: abs(x - median_sum_len))
    df = df.nsmallest(n=50, columns='abs_distance')

    print('Loaded {} predictions.'.format(df.shape[0]))
    predictions = list(map(prepare_str_for_rouge, df['prediction'].tolist()))
    references = list(map(prepare_str_for_rouge, df['reference'].tolist()))
    n = len(df)
    rouge_types = ['rouge1', 'rouge2']
    outputs = compute(predictions, references, rouge_types=rouge_types, use_aggregator=False)
    f1_means = np.array(
        [sum([outputs[t][i].fmeasure for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    recall_means = np.array(
        [sum([outputs[t][i].recall for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    precision_means = np.array(
        [sum([outputs[t][i].precision for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])

    df['f1'] = f1_means
    df['recall'] = recall_means
    df['precision'] = precision_means

    out_fn = os.path.join(out_dir, 'human', 'oracle_greedy_rel_hiv_validation_scored.csv')
    print('Saving {} ROUGE-scored examples to {}'.format(len(df), out_fn))
    df.to_csv(out_fn, index=False)

    expanded_df = []
    for record in df.to_dict('records'):
        mrn = record['mrn']
        account = record['account']
        visit_start = record['admit_date_min']
        visit_end = record['discharge_date_max']

        expanded_df.append({
            'visit_id': account,
            'mrn': mrn,
            'visit_start': visit_start,
            'visit_end': visit_end,
            'hospital_course': record['reference'],
            'is_ref': 1,
        })

        expanded_df.append({
            'visit_id': account,
            'mrn': mrn,
            'visit_start': visit_start,
            'visit_end': visit_end,
            'hospital_course': record['prediction'],
            'is_ref': 0
        })

    out_fn = os.path.join(out_dir, 'human', 'form_data_flat.csv')
    expanded_df = pd.DataFrame(expanded_df)
    expanded_df.to_csv(out_fn, index=False)
