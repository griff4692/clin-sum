import json
import sys

import pandas as pd


if __name__ == '__main__':
    i = int(sys.argv[1])
    df = pd.read_csv('/nlp/projects/clinsum/srl_packed_examples_small.csv')

    for record in df[i:].to_dict('records'):
        toks = record['flat_spacy_target_toks']
        labels = record['cui_label_names']
        srl_target = record['srl_packed_target']

        all_toks = json.loads(toks)
        print(' '.join(all_toks))
        print('MRN={}. Account={}.'.format(record['mrn'], record['account']))
        break
        all_labels = json.loads(labels)
        assert len(all_toks) == len(all_labels)

        viz = ['<START:{}>'.format(all_labels[0])]
        for i, (t, l) in enumerate(zip(all_toks, all_labels)):
            if i > 0 and not all_labels[i - 1] == all_labels[i]:
                viz += ['<END:{}>'.format(all_labels[i - 1]), '<START:{}>'.format(all_labels[i])]
            viz += [t]
        viz += ['<END:{}>'.format(all_labels[-1])]
        print(' '.join(viz))
        print('\n')
        print(srl_target)
        print('\n\n')

        break