import os

import argparse
from collections import defaultdict
from nlp import load_metric
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ROUGE Scorer')
    parser.add_argument('--in_fn', default='../extractive/predictions/lr_validation.csv')

    args = parser.parse_args()

    metric = load_metric('rouge')

    df = pd.read_csv(args.in_fn)
    predictions = df['prediction'].tolist()[:10]
    references = df['reference'].tolist()[:10]
    rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    scores = metric.compute(predictions=predictions, references=references, rouge_types=rouge_types)

    results = []
    stats = ['low', 'mid', 'high']
    measures = ['precision', 'recall', 'fmeasure']
    print('Showing mean statistics...')
    for name, agg_score in scores.items():
        print(name)
        for stat in stats:
            score_obj = getattr(agg_score, stat)
            for measure in measures:
                value = getattr(score_obj, measure)
                results.append({
                    'name': name,
                    'measure': measure,
                    'agg_type': stat,
                    'value': value
                })
                if stat == 'mid':
                    print('\t{}={}'.format(measure, value))
        print('\n')
    results_df = pd.DataFrame(results)

    out_fn = os.path.join('results', args.in_fn.split('/')[-1])
    print('Saving results to {}'.format(out_fn))
    results_df.to_csv(out_fn, index=False)
