from collections import defaultdict, Counter
import itertools
from functools import partial
import os
import re
import string
import sys
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_uimap
import spacy

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.section_utils import resolve_course, sent_toks_from_html
from preprocess.utils import *


if __name__ == '__main__':
    _, _, mrns = get_mrn_status_df('valid_example')

    mrn_sample = np.random.choice(mrns, size=500, replace=False)
    target_lens = []
    source_lens = []

    for i in tqdm(range(len(mrn_sample))):
        mrn = mrn_sample[i]
        mrn_dir = os.path.join(out_dir, 'mrn', mrn)
        examples_fn = os.path.join(mrn_dir, 'examples.csv')
        examples_df = pd.read_csv(examples_fn)

        for example in examples_df.to_dict('records'):
            target_lens.append(len(sent_toks_from_html(resolve_course(example['spacy_target_toks']))))
            source_lens.append(len(sent_toks_from_html(example['spacy_source_toks'])))

    df = pd.DataFrame({'target': target_lens, 'source': source_lens})
    df.to_csv('tmp.csv', index=False)