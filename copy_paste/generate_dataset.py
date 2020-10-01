import itertools
import os
import sys

import argparse
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.utils import *
from preprocess.section_utils import *


def gen_examples(example):
    mrn = example['mrn']
    account = example['account']
    split_text = re.split(HTML_REGEX, example['srl_packed_source'])
    is_tag_full = list(map(lambda x: re.search(HTML_REGEX, x) is not None, split_text))
    examples = []
    curr_note_id = None
    chunk_idx = 0
    curr_chunk = ''
    curr_header = None
    curr_state = None
    for i, str in enumerate(split_text):
        str = str.strip()
        if len(str) == 0:
            continue

        is_tag = is_tag_full[i]
        if is_tag:
            if str == '</p>':
                assert len(curr_chunk) > 0
                examples.append({
                    'mrn': mrn,
                    'account': account,
                    'note_id': curr_note_id,
                    'chunk_idx': chunk_idx,
                    'header': curr_header,
                    'chunk': curr_chunk,
                    'has_frag': '<f>' in curr_chunk
                })
                chunk_idx += 1
                curr_chunk = ''
            elif str[1] == '/':
                continue
            else:
                curr_state = str[1]
                assert curr_state in {'p', 'c', 'd', 'e', 'u', 'f', 'h'}
                if curr_state == 'd':
                    curr_note_id = str[11:-1]
                    chunk_idx = 0
        else:
            if curr_state in {'p', 'u'}:
                curr_chunk += ' ' + str + ' '
            elif curr_state == 'f':
                curr_chunk += ' <f> ' + str + ' </f>'
            elif curr_state == 'h':
                curr_header = str.strip()
            else:
                print(str)
                print(split_text[i - 1])
                raise Exception('Unknown state: {}'.format(curr_state))

    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to pack source and summary with entity and copy-paste fragments.')
    parser.add_argument('-mini', default=False, action='store_true')

    args = parser.parse_args()

    mini_str = '_small' if args.mini else ''

    in_fn = os.path.join(out_dir, 'srl_packed_examples{}.csv'.format(mini_str))
    out_fn = os.path.join(out_dir, 'copy_paste_dataset{}.csv'.format(mini_str))
    print('Reading in srl packed examples from {}'.format(in_fn))
    df = pd.read_csv(in_fn)
    n = len(df)
    print('Generating copy-paste example chunks for {} examples...'.format(n))
    outputs = list(tqdm(map(gen_examples, df.to_dict('records')), total=n))
    outputs_flat = list(itertools.chain(*outputs))
    print('Extracted {} chunks'.format(len(outputs_flat)))

    df = pd.DataFrame(outputs_flat)
    df.drop_duplicates(subset=['chunk'], inplace=True)
    df_w_frag = df[df['has_frag']]
    df_wout_frag = df[~df['has_frag']]
    num_w_frag = len(df_w_frag)
    print('{} examples are positive - i.e., have at least 1 fragment'.format(num_w_frag))
    target_num_wout = 2 * num_w_frag
    df_wout_frag_downsampled = df_wout_frag.sample(n=min(target_num_wout, len(df_wout_frag)), replace=False)
    downsampled_df = pd.concat([df_w_frag, df_wout_frag_downsampled])
    print('Saving {} chunks to {}'.format(len(downsampled_df), out_fn))
    downsampled_df.to_csv(out_fn, index=False)
