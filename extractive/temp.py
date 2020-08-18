import os

import pandas as pd


if __name__ == '__main__':
    pre_dir = '/nlp/projects/clinsum/predictions'
    type = 'oracle'
    fn = os.path.join(pre_dir, '{}_validation.csv'.format(type))
    df = pd.read_csv(fn)
    records = df.to_dict('records')
    for record in records:
        # print(record['prediction'])
        # print('\n')
        # print(record['reference'])
        pred_sents = record['prediction'].split('<s> ')
        target_sents = record['reference'].split('<s> ')
        print('Target:')
        for t in target_sents:
            print('\t' + t)

        print('Predicted:')
        for p in pred_sents:
            print('\t' + p)
        print('\n')

        print('\n\n')
