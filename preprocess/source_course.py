import itertools
import os
import re
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess.constants import *
from preprocess.utils import *
from preprocess.section_utils import section_split


if __name__ == '__main__':
    df = pd.read_csv('/nlp/projects/clinsum/full_examples_small.csv')
    n = len(df)
    num_zero = 0
    for record in df.to_dict('records'):
        st = record['source_str']
        sectioned_text, is_header_arr = section_split(st)
        is_relevant_section = list(map(
            lambda x: x[0] and x[1] is not None and 'course' in x[1].lower() and len(x[1]) < MAX_HEADER_LEN,
            zip(is_header_arr, sectioned_text)
        ))
        num_relevant = len([ir for ir in is_relevant_section if ir])
        if num_relevant == 0:
            num_zero += 1
        else:
            print(record['mrn'], record['account'])

    print('{} out of {} are OK. {} percent'.format(
        num_zero, n, num_zero / float(n)
    ))
