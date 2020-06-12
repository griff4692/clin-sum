import os
import sys

import pandas as pd

delim = '\n' + '-' * 100 + '\n'


if __name__ == '__main__':
    hadm_id = sys.argv[1]
    visit_dir = os.path.join('visits', str(hadm_id))

    dsum_df = pd.read_csv(os.path.join(visit_dir, 'dsum.csv'))

    rel_cols = [c for c in dsum_df.columns if not c == 'TEXT']

    with open('tmp.txt', 'w') as fd:
        fd.write(','.join(rel_cols) + '\n')
        for i, row in dsum_df.iterrows():
            row = row.to_dict()
            vals = [str(row[k]) for k in rel_cols]
            fd.write(','.join(vals) + '\n')

        fd.write(delim)
        fd.write(delim.join(dsum_df['TEXT'].tolist()))
