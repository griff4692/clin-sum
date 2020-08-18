import os

import numpy as np
import pandas as pd

from constants import *
from utils import *


if __name__ == '__main__':
    _, _, mrns = get_mrn_status_df('valid_example')
    n = len(mrns)
    sample_n = min(n, 25)
    mrn_sample = np.random.choice(mrns, size=sample_n, replace=False)
    dfs = []
    for mrn in mrn_sample:
        mrn_dir = os.path.join(out_dir, 'mrn', str(mrn))
        examples_df = pd.read_csv(os.path.join(mrn_dir, 'examples.csv'))
        dfs.append(examples_df)

    dfs = pd.concat(dfs)
    out_fn = os.path.join(out_dir, 'sample_examples.csv')
    dfs.to_csv(out_fn, index=False)